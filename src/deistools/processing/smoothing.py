"""
Smoothing of the time series that come out of a DEIS analysis -- e.g. the impedance at one frequency over time --
and removal of isolated outliers. Pure numpy (scipy is not needed; the tests compare against it when it is installed).

All filters assume evenly spaced samples (the blocks of the online FFT-EIS and the points of DMFA are), work along one
axis of an array of any shape, and accept real or complex data (real and imaginary parts are filtered separately, or
magnitude and unwrapped phase with `smooth_impedance(..., representation="polar")`). Three things make them safe to
use on measured series:

* `segments`: the series of a run that had several techniques is the concatenation of independent pieces whose
  levels differ (a current step, an OCV rest). Pass the boundaries and every piece is filtered on its own, so a step
  stays a step instead of being blurred across the boundary. `segment_bounds(counts)` builds them from the number
  of points of each piece.
* NaN values (a missing point) are bridged by linear interpolation for the filtering and put back afterwards: a gap
  stays a gap and does not spread.
* a piece shorter than the window gets the largest odd window that fits (a piece with fewer than 3 points is
  returned unchanged), never an error.

Methods (METHODS): Savitzky-Golay (keeps peaks and curvature; also gives derivatives), Gaussian (smooth, no ringing),
moving average, moving median (robust to spikes). `hampel` replaces isolated outliers by the local median.
"""
from math import factorial

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

METHOD_NONE = "None"
METHOD_SAVGOL = "Savitzky-Golay"
METHOD_GAUSSIAN = "Gaussian"
METHOD_MEAN = "Moving average"
METHOD_MEDIAN = "Moving median"
METHODS = (METHOD_NONE, METHOD_SAVGOL, METHOD_GAUSSIAN, METHOD_MEAN, METHOD_MEDIAN)

EDGE_NEAREST = "nearest"   # beyond the ends the first/last value is repeated
EDGE_MIRROR = "mirror"     # the series is mirrored about its end points
EDGE_ODD = "odd"           # the series continues with the slope it has at the end (point reflection): no bias on a drift
EDGE_INTERP = "interp"     # Savitzky-Golay only: a polynomial is fitted to the last window and evaluated there
EDGE_MODES = (EDGE_NEAREST, EDGE_MIRROR, EDGE_ODD, EDGE_INTERP)

REPRESENTATION_CARTESIAN = "cartesian"   # real and imaginary part
REPRESENTATION_POLAR = "polar"           # magnitude and unwrapped phase

_NUMPY_PAD = {EDGE_NEAREST: dict(mode="edge"), EDGE_MIRROR: dict(mode="reflect"),
              EDGE_ODD: dict(mode="reflect", reflect_type="odd")}


# ---------------------------------------------------------------------------------------------- helpers
def odd_window(points, minimum=3):
    """`points` rounded to the nearest odd integer, at least `minimum` (a filter window must be odd)."""
    points = int(round(float(points)))
    if points % 2 == 0:
        points += 1
    return max(points, minimum)


def window_points(seconds, dt, minimum=3):
    """Odd number of samples that covers `seconds` for samples `dt` seconds apart."""
    return odd_window(seconds / dt, minimum)


def segment_bounds(counts):
    """Indices where each piece of a concatenated series starts, from the number of points of each piece
    (the first start, 0, and the total are not included): [4, 6, 5] -> [4, 10]."""
    counts = np.asarray(counts, dtype=int)
    return np.cumsum(counts)[:-1] if counts.size > 1 else np.array([], dtype=int)


def _segment_slices(n, segments):
    if segments is None:
        return [slice(0, n)]
    cuts = sorted({int(b) for b in np.asarray(segments).ravel() if 0 < int(b) < n})
    edges = [0, *cuts, n]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _fill_nans(y):
    """Linear interpolation over the NaNs of a 1-D float array. Returns (filled, nan_mask)."""
    mask = ~np.isfinite(y)
    if not mask.any():
        return y, mask
    if mask.all():
        return y, mask
    x = np.arange(y.size)
    filled = y.copy()
    filled[mask] = np.interp(x[mask], x[~mask], y[~mask])
    return filled, mask


def _pad(y, half, edge):
    return np.pad(y, half, **_NUMPY_PAD[edge])


def _fit_window(length, window, minimum=3):
    """Largest odd window <= `window` that fits in `length` points (None if there is no usable one)."""
    window = min(window, length if length % 2 else length - 1)
    return window if window >= minimum else None


# ---------------------------------------------------------------------------------------------- Savitzky-Golay
def savgol_coefficients(window, polyorder, deriv=0, delta=1.0):
    """Coefficients c[k], k = -m..m (window = 2m + 1), such that sum_k c[k] y[i + k] is the `deriv`-th derivative
    (0 = the smoothed value) at i of the least-squares polynomial of order `polyorder` through the window.
    `delta` is the spacing of the samples (only used by derivatives)."""
    if window % 2 == 0 or window < 1:
        raise ValueError("the window must be a positive odd number of points")
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than the window")
    if deriv > polyorder:
        return np.zeros(window)
    m = window // 2
    x = np.arange(-m, m + 1, dtype=float)
    pseudo_inverse = np.linalg.pinv(np.vander(x, polyorder + 1, increasing=True))
    return pseudo_inverse[deriv] * factorial(deriv) / delta ** deriv


def _savgol_1d(y, window, polyorder, deriv, delta, edge):
    n = y.size
    window = _fit_window(n, window, minimum=polyorder + 1 if (polyorder + 1) % 2 else polyorder + 2)
    if window is None:
        return y.copy()
    m = window // 2
    c = savgol_coefficients(window, polyorder, deriv, delta)
    if edge == EDGE_INTERP:
        out = np.empty(n)
        out[m:n - m] = np.correlate(y, c, mode="valid")
        x = np.arange(window, dtype=float)
        for first, sl, where in ((0, slice(0, window), x[:m]), (n - m, slice(n - window, n), x[window - m:])):
            coefficients = np.polyfit(x, y[sl], polyorder)
            if deriv:
                coefficients = np.polyder(coefficients, deriv) if deriv <= polyorder else np.zeros(1)
            out[first:first + m] = np.polyval(coefficients, where) / delta ** deriv
        return out
    return np.correlate(_pad(y, m, edge), c, mode="valid")


# ---------------------------------------------------------------------------------------------- other kernels
def _gaussian_kernel(sigma):
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _kernel_1d(y, kernel, edge):
    half = kernel.size // 2
    if y.size < 3:
        return y.copy()
    return np.correlate(_pad(y, half, edge), kernel, mode="valid")


def _median_1d(y, window, edge):
    window = _fit_window(y.size, window)
    if window is None:
        return y.copy()
    half = window // 2
    return np.median(sliding_window_view(_pad(y, half, edge), window), axis=-1)


# ---------------------------------------------------------------------------------------------- public: smooth
def _smooth_real_1d(y, method, window, polyorder, sigma, deriv, delta, edge):
    if method == METHOD_SAVGOL:
        return _savgol_1d(y, window, polyorder, deriv, delta, edge)
    if edge == EDGE_INTERP:
        edge = EDGE_NEAREST          # the polynomial extrapolation only exists for Savitzky-Golay
    if method == METHOD_GAUSSIAN:
        return _kernel_1d(y, _gaussian_kernel(sigma), edge) if sigma > 0 else y.copy()
    if method == METHOD_MEAN:
        window = _fit_window(y.size, window)
        return y.copy() if window is None else _kernel_1d(y, np.full(window, 1.0 / window), edge)
    if method == METHOD_MEDIAN:
        return _median_1d(y, window, edge)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def _along_last_axis(y, function):
    """Applies function(1-D float array) -> 1-D array to every row of the last axis."""
    flat = y.reshape(-1, y.shape[-1])
    return np.array([function(row) for row in flat]).reshape(y.shape)


def smooth(y, method=METHOD_SAVGOL, *, window=11, polyorder=2, sigma=2.0, deriv=0, delta=1.0,
           edge=EDGE_NEAREST, segments=None, axis=-1):
    """
    Smooths `y` along `axis`. Real or complex (real and imaginary parts separately); any shape.

    method     one of METHODS (METHOD_NONE returns a copy).
    window     points, used by Savitzky-Golay, moving average and moving median (made odd; a piece shorter than
               the window gets the largest odd one that fits).
    polyorder  Savitzky-Golay polynomial order (< window).
    sigma      Gaussian standard deviation in points (the kernel is cut at 4 sigma).
    deriv, delta  Savitzky-Golay derivative order and sample spacing (the result is dy/dt**deriv).
    edge       what is assumed beyond the two ends: EDGE_NEAREST (flat), EDGE_MIRROR, EDGE_ODD (same slope) or
               (Savitzky-Golay only) EDGE_INTERP.
    segments   indices where an independent piece starts (see segment_bounds); each piece is filtered alone.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if edge not in EDGE_MODES:
        raise ValueError(f"unknown edge mode {edge!r}; expected one of {EDGE_MODES}")
    y = np.asarray(y)
    if method == METHOD_NONE:
        return y.copy()
    if np.iscomplexobj(y):
        kw = dict(window=window, polyorder=polyorder, sigma=sigma, deriv=deriv, delta=delta, edge=edge,
                  segments=segments, axis=axis)
        return smooth(y.real, method, **kw) + 1j * smooth(y.imag, method, **kw)
    moved = np.moveaxis(np.asarray(y, dtype=float), axis, -1)
    slices = _segment_slices(moved.shape[-1], segments)
    window = odd_window(window, minimum=1)
    if method == METHOD_SAVGOL and polyorder >= window:
        raise ValueError(f"the Savitzky-Golay polynomial order ({polyorder}) must be smaller than the window ({window})")

    def one_row(row):
        out = np.empty_like(row)
        for sl in slices:
            piece = row[sl]
            filled, nan_mask = _fill_nans(piece)
            if nan_mask.all():
                out[sl] = piece
                continue
            result = _smooth_real_1d(filled, method, window, polyorder, sigma, deriv, delta, edge)
            result[nan_mask] = np.nan
            out[sl] = result
        return out

    return np.moveaxis(_along_last_axis(moved, one_row), -1, axis)


def smooth_impedance(z, method=METHOD_SAVGOL, *, representation=REPRESENTATION_CARTESIAN, segments=None,
                     axis=-1, **kwargs):
    """
    Smooths a complex impedance array along `axis` (time). `representation` chooses what is filtered:
    REPRESENTATION_CARTESIAN: Z' and Z'' (what a Nyquist plot shows); REPRESENTATION_POLAR: |Z| and the
    unwrapped phase (what a Bode plot shows; a phase that wraps around +-pi is handled). The other arguments are
    those of `smooth`.
    """
    z = np.asarray(z, dtype=complex)
    if method == METHOD_NONE:
        return z.copy()
    if representation == REPRESENTATION_CARTESIAN:
        return smooth(z, method, segments=segments, axis=axis, **kwargs)
    if representation != REPRESENTATION_POLAR:
        raise ValueError(f"representation must be {REPRESENTATION_CARTESIAN!r} or {REPRESENTATION_POLAR!r}")
    moved = np.moveaxis(z, axis, -1)
    slices = _segment_slices(moved.shape[-1], segments)
    phase = np.empty(moved.shape, dtype=float)
    for index in np.ndindex(moved.shape[:-1]):
        row = moved[index]
        raw = np.angle(row)
        for sl in slices:
            filled, nan_mask = _fill_nans(raw[sl])
            if not nan_mask.all():
                filled = np.unwrap(filled)
            filled[nan_mask] = raw[sl][nan_mask]
            phase[index][sl] = filled
    magnitude = smooth(np.abs(moved), method, segments=segments, axis=-1, **kwargs)
    phase = smooth(phase, method, segments=segments, axis=-1, **kwargs)
    return np.moveaxis(magnitude * np.exp(1j * phase), -1, axis)


# ---------------------------------------------------------------------------------------------- outliers
def hampel(y, window=7, n_sigmas=3.0, *, edge=EDGE_ODD, segments=None, axis=-1):
    """
    Hampel filter: a point whose distance from the local median exceeds `n_sigmas` times the local scatter
    (1.4826 * median absolute deviation) is an outlier and is replaced by the median. Complex data are judged by
    the distance of the point from the median of the real and of the imaginary part, and replaced as a whole.
    Returns (cleaned, outlier_mask). `window`, `edge` (default EDGE_ODD: a drifting series is not mistaken for outliers at its ends), `segments`
    and `axis` as in `smooth`.
    """
    if edge == EDGE_INTERP:
        edge = EDGE_NEAREST
    y = np.asarray(y)
    complex_input = np.iscomplexobj(y)
    moved = np.moveaxis(y.astype(complex if complex_input else float), axis, -1)
    slices = _segment_slices(moved.shape[-1], segments)
    window = odd_window(window)
    cleaned = moved.copy()
    mask = np.zeros(moved.shape, dtype=bool)

    def windows(values, size):
        return sliding_window_view(_pad(values, size // 2, edge), size)

    for index in np.ndindex(moved.shape[:-1]):
        row = moved[index]
        for sl in slices:
            piece = row[sl]
            size = _fit_window(piece.size, window)
            if size is None:
                continue
            real, imag = piece.real.astype(float), piece.imag.astype(float)
            real, bad_r = _fill_nans(real)
            imag, bad_i = _fill_nans(imag)
            win_r = windows(real, size)
            med_r = np.median(win_r, axis=-1)
            if complex_input:
                win_i = windows(imag, size)
                med_i = np.median(win_i, axis=-1)
                spread = np.hypot(win_r - med_r[:, None], win_i - med_i[:, None])
            else:
                med_i = np.zeros_like(med_r)
                spread = np.abs(win_r - med_r[:, None])
            distance = np.hypot(real - med_r, imag - med_i)
            scatter = 1.4826 * np.median(spread, axis=-1)   # median absolute deviation of the window
            tolerance = 1e-9 * np.hypot(med_r, med_i)   # rounding noise on a flat series is not an outlier
            outlier = (distance > n_sigmas * scatter + tolerance) & ~(bad_r | bad_i)
            replacement = med_r + 1j * med_i if complex_input else med_r
            part = cleaned[index][sl]
            part[outlier] = replacement[outlier]
            cleaned[index][sl] = part
            mask[index][sl] = outlier
    return np.moveaxis(cleaned, -1, axis), np.moveaxis(mask, -1, axis)
