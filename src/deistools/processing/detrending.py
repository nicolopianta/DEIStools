import numpy as np

# def compute_baseline(signal):
    
#     xstart = 0
#     ystart = signal[0]
#     xfinish = signal.size
#     yfinish = signal[-1]
    
#     m = (yfinish - ystart) / (xfinish - xstart)
#     q = yfinish - (m * xfinish)
    
#     line = np.arange(xstart, xfinish, 1) * m + q 
    
#     return m, q,


# def remove_baseline(signal, m, q):
#     line = np.arange(signal[0], signal[-1], 1) * m + q 
#     return signal - line


# def redo_baseline(signal, m, q):
#     line = np.arange(signal[0], signal[-1], 1) * m + q   
#     return signal + line

#------------------------------------------------------------------------------#

def remove_baseline(signal, sampling_time):
    'Tested on 22.03 working preperly'
    coordinates = {'xstart'  : 0,
                   'ystart'  : signal[0],
                   'xfinish' : signal.size * sampling_time,
                   'yfinish' : signal[-1]
                   }    
    m = (coordinates['yfinish'] - coordinates['ystart']) / (coordinates['xfinish'] - coordinates['xstart'])
    q = coordinates['yfinish'] - (m * coordinates['xfinish'])
    line = np.arange(coordinates['xstart'], coordinates['xfinish'], sampling_time) * m + q 
    signal = signal - line   
    return signal, coordinates

#------------------------------------------------------------------------------#

def redo_baseline(signal, coordinates): 
    'Tested on 22.03 working preperly'
    m = (coordinates['yfinish'] - coordinates['ystart']) / (coordinates['xfinish'] - coordinates['xstart'])
    q = coordinates['yfinish'] - (m * coordinates['xfinish'])
    line = np.linspace(coordinates['xstart'], coordinates['xfinish'], signal.size) * m + q  
    signal = signal + line  
    return signal

#------------------------------------------------------------------------------#
# Trend removal before the FFT of a long record (e.g. for DMFA).
#
# DMFA Fourier-transforms the whole record at once. A record is not periodic:
# the potential typically drifts by volts between its first and last sample, and
# that edge jump leaks, roughly as 1/k, into every FFT bin -- including the
# lowest multisine tones -- ruining the first and last DMFA points. Removing a
# slow trend first removes that leakage. The trend is returned as well, because
# it is the physical DC part of the signal and has to be added back to the
# zero-frequency (DC) result.
#
# Two things make a naive detrend (e.g. a least-squares line) unsafe:
#  * the lowest tones correlate with a ramp, so a plain fit would absorb part of
#    the signal. The fits are therefore made on block means taken over exactly one
#    period of the lowest tone (which cancels every tone that is an integer
#    multiple of it -- true for the usual designs), and the polynomial/spline
#    methods iterate: estimate the tones, subtract them, refit the trend, repeat.
#  * a mirrored (even) extension of the record, the usual way to kill an edge
#    jump, scrambles the tones' phase, so it is deliberately not offered.

METHOD_NONE = "None"
METHOD_LINE = "Linear (end points)"
METHOD_POLY = "Polynomial (iterative)"
METHOD_SPLINE = "Smooth spline (iterative)"
METHODS = (METHOD_NONE, METHOD_LINE, METHOD_POLY, METHOD_SPLINE)


def block_means(x, block):
    """Means of consecutive `block`-sample blocks (the last, partial block
    included) and their centre sample positions."""
    starts = np.arange(0, x.size, block)
    sums = np.add.reduceat(x, starts)
    counts = np.diff(np.append(starts, x.size))
    return sums / counts, starts + (counts - 1) / 2.0


def endpoint_line(x, block):
    """Straight line through the mean of the first and of the last `block`
    samples (each assigned to its window centre), evaluated on every sample."""
    n = x.size
    block = max(1, min(block, n // 2))
    y0, y1 = x[:block].mean(), x[-block:].mean()
    c0, c1 = (block - 1) / 2.0, n - 1 - (block - 1) / 2.0
    slope = (y1 - y0) / (c1 - c0)
    return y0 + slope * (np.arange(n) - c0)


def tone_signal(x, fs, freqs, half_bw):
    """Time-domain sum of the narrow bands (+-half_bw Hz) around each of
    `freqs` in `x` -- an estimate of the multisine tones, DC excluded."""
    n = x.size
    spectrum = np.fft.rfft(x)
    axis = np.fft.rfftfreq(n, 1.0 / fs)
    mask = np.zeros(axis.size, dtype=bool)
    df = fs / n
    half_bins = max(1, int(np.ceil(half_bw / df)))
    centres = np.round(np.asarray(freqs, dtype=float) / df).astype(int)
    for c in centres:
        mask[max(c - half_bins, 1):min(c + half_bins + 1, axis.size)] = True
    spectrum[~mask] = 0.0
    return np.fft.irfft(spectrum, n)


def fit_smooth_trend(x, block, method, order, knot_spacing_blocks):
    """Polynomial (Legendre) or cubic-spline (least squares, uniform knots) fit through
    the block means of `x`, evaluated on every sample. The spline needs scipy."""
    yc, cc = block_means(x, block)
    n = x.size
    t = np.arange(n, dtype=float)
    if method == METHOD_POLY:
        scale = max(n - 1, 1) / 2.0
        coeffs = np.polynomial.legendre.legfit((cc - scale) / scale, yc, order)
        return np.polynomial.legendre.legval((t - scale) / scale, coeffs)
    try:
        from scipy.interpolate import LSQUnivariateSpline
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The spline detrend needs scipy (pip install scipy).") from exc

    step = max(1, int(round(knot_spacing_blocks)))
    knots = cc[step::step]
    knots = knots[(knots > cc[1]) & (knots < cc[-2])] if cc.size > 3 else knots[:0]
    return LSQUnivariateSpline(cc, yc, knots, k=3)(t)


def detrend(x, fs, freqs, method, order=3, knot_spacing_periods=10.0, n_iter=3, half_bw=None):
    """
    Returns (detrended, trend) with detrended + trend == x.

    x                     samples, uniformly sampled at fs [Hz]
    freqs                 the multisine frequencies present in x [Hz]; the lowest
                          one sets the averaging block (one period of it)
    method                one of METHODS
    order                 polynomial order (METHOD_POLY)
    knot_spacing_periods  spline knot spacing, in periods of the lowest frequency
    n_iter                tone-subtraction/refit iterations (poly and spline)
    half_bw               half-width [Hz] of the band taken as "tone" around each
                          frequency when estimating them; defaults to 0.1 x lowest
    """
    x = np.asarray(x, dtype=float)
    if method == METHOD_NONE or x.size < 4:
        return x.copy(), np.zeros_like(x)
    freqs = np.asarray(freqs, dtype=float)
    f_min = float(freqs.min())
    block = max(1, int(round(fs / f_min)))
    if method == METHOD_LINE:
        trend = endpoint_line(x, block)
        return x - trend, trend
    if x.size < 4 * block:
        raise ValueError(
            f"Record ({x.size / fs:.0f} s) is shorter than 4 periods of the lowest frequency "
            f"({4 / f_min:.0f} s) -- too short to fit a trend."
        )
    if half_bw is None:
        half_bw = 0.1 * f_min
    trend = fit_smooth_trend(x, block, method, order, knot_spacing_periods)
    for _ in range(max(0, n_iter)):
        tones = tone_signal(x - trend, fs, freqs, half_bw)
        trend = fit_smooth_trend(x - tones, block, method, order, knot_spacing_periods)
    return x - trend, trend
