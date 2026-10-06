"""
DMFA on a whole decimated voltage/current record, end to end: optional detrend,
FFT, peak search, time-resolved impedance at each frequency and time-resolved DC
voltage/current (with the removed trend put back).
"""
from dataclasses import dataclass, field

import numpy as np

from deistools.processing import detrending
from deistools.processing.dmfa_functions import block_average
from deistools.processing.mfa import MultiFrequencyAnalysis


@dataclass
class DmfaResult:
    frequencies: np.ndarray        # (n_freq,) Hz, as given
    impedance: np.ndarray          # (n_freq, n_points) complex
    zero_voltage: np.ndarray       # (n_points,) V, DC part (trend included)
    zero_current: np.ndarray       # (n_points,) A, DC part (trend included)
    time: np.ndarray               # (n_points,) s, centre of each point
    freq_indexes: list             # FFT bins found for each frequency (fftshift-ed axis)
    block_len: int                 # samples averaged into one output point
    n_samples: int                 # samples used (a multiple of block_len)
    detrend_method: str
    voltage_trend: np.ndarray = field(default=None, repr=False)   # only with keep_trends=True
    current_trend: np.ndarray = field(default=None, repr=False)


def run_dmfa_on_record(
        voltage,
        current,
        frequencies,
        fs,
        *,
        bw_zero_mult=0.8,
        bw_filter_mult=0.1,
        zero_filter_order=100,
        filter_order=8,
        seconds_per_point=20.0,
        detrend_method=detrending.METHOD_NONE,
        detrend_order=3,
        knot_spacing_periods=3.0,
        keep_trends=False,
    ) -> DmfaResult:
    """
    voltage, current      decimated record, uniformly sampled at `fs` [Hz]
    frequencies           multisine frequencies to analyse [Hz] (the low band)
    bw_zero_mult          DC low-pass bandwidth, in multiples of the lowest frequency
    bw_filter_mult        per-tone band-pass bandwidth, in multiples of the lowest frequency
    zero_filter_order,
    filter_order          Fermi-Dirac steepness for the DC / tone filters
    seconds_per_point     time resolution of the output (the record is trimmed to a whole
                          number of these blocks)
    detrend_method, ...   see deistools.processing.detrending.detrend
    """
    frequencies = np.asarray(frequencies, dtype=float)
    sampling_time = 1.0 / fs
    n_raw = min(np.size(voltage), np.size(current))
    block_len = max(1, int(round(seconds_per_point / sampling_time)))
    n = (n_raw // block_len) * block_len
    if n == 0:
        raise ValueError(
            f"Not enough data ({n_raw} samples) for even one {seconds_per_point:g} s DMFA block."
        )
    npts = n // block_len
    f_min = float(frequencies.min())

    kwargs = dict(order=detrend_order, knot_spacing_periods=knot_spacing_periods, half_bw=bw_filter_mult * f_min)
    voltage_in, voltage_trend = detrending.detrend(np.asarray(voltage)[:n], fs, frequencies, detrend_method, **kwargs)
    current_in, current_trend = detrending.detrend(np.asarray(current)[:n], fs, frequencies, detrend_method, **kwargs)

    mfa = MultiFrequencyAnalysis(frequencies, voltage_in, current_in, sampling_time)
    mfa.compute_freq_axis()
    mfa.compute_fft()
    mfa.search_freq_indexes(mfa.ft_current)
    impedance, zero_voltage, zero_current, time = mfa.run_dmfa_fd(
        time_resolution=sampling_time,
        bw_zero=bw_zero_mult * f_min,
        bw_filter=bw_filter_mult * f_min,
        filter_order=filter_order,
        Npts_elab=npts,
        zero_filter_order=zero_filter_order,
    )
    # The trend taken out before the FFT is the physical DC part of the signal: put it
    # back, block-averaged exactly as run_dmfa_fd averages its own output.
    zero_voltage = zero_voltage + block_average(voltage_trend, block_len)
    zero_current = zero_current + block_average(current_trend, block_len)
    return DmfaResult(
        frequencies=frequencies,
        impedance=impedance,
        zero_voltage=zero_voltage,
        zero_current=zero_current,
        time=np.asarray(time),
        freq_indexes=list(mfa.freq_indexes),
        block_len=block_len,
        n_samples=n,
        detrend_method=detrend_method,
        voltage_trend=voltage_trend if keep_trends else None,
        current_trend=current_trend if keep_trends else None,
    )


def zero_order_hold_resample(rows, row_times, new_times):
    """
    Resample `rows` (n_rows, n_old), whose columns sit at `row_times` (ascending),
    onto `new_times` by holding each column's value until the next one: for every new
    time the latest column at or before it (the first column for earlier times).
    """
    rows = np.asarray(rows)
    idx = np.clip(np.searchsorted(row_times, new_times, side="right") - 1, 0, rows.shape[1] - 1)
    return rows[:, idx]
