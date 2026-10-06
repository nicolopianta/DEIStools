"""Small spectrum helpers (pure numpy)."""
import numpy as np


def amplitude_spectrum(x, fs):
    """Single-sided amplitude spectrum of `x` (DC bin dropped): returns
    (frequencies [Hz], amplitude in the units of x -- a sine of amplitude A
    shows up as A at its frequency)."""
    x = np.asarray(x, dtype=float)
    amplitude = 2.0 * np.abs(np.fft.rfft(x)) / x.size
    return np.fft.rfftfreq(x.size, 1.0 / fs)[1:], amplitude[1:]


def log_binned_max(freqs, amplitude, n_bins=3000):
    """
    Display reduction of a long linear-frequency spectrum for a log-x plot:
    keeps every bin at the low end (where there are fewer than one FFT bin
    per log-bin) and the maximum of each geometric bin above that, so narrow
    tones survive. Returns (x, y).
    """
    n = amplitude.size
    if n <= n_bins:
        return freqs, amplitude
    edges = np.unique(np.floor(np.geomspace(1, n, n_bins)).astype(int) - 1)
    edges = np.append(edges[edges < n], n)
    y = np.maximum.reduceat(amplitude, edges[:-1])
    x = np.sqrt(freqs[edges[:-1]] * freqs[edges[1:] - 1])  # geometric centre of each bin
    return x, y
