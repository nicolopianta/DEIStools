import unittest

import numpy as np

from deistools.processing import MultiFrequencyAnalysis


def _mfa(freqs, fs, n, voltage, current):
    m = MultiFrequencyAnalysis(np.asarray(freqs, dtype=float), voltage, current, 1 / fs)
    m.compute_freq_axis()
    m.compute_fft()
    return m


class SearchFreqIndexes(unittest.TestCase):
    def test_one_period_window_never_picks_the_dc_bin(self):
        # window = exactly one period of the lowest tone: that tone sits at bin 1, right
        # next to DC, whose content (here a large offset) is far bigger than the tone
        fs, n = 100.0, 200
        t = np.arange(n) / fs
        freqs = [0.5, 1.5, 3.0]
        current = 3.0 + 0.003 * t + sum(0.01 * np.cos(2 * np.pi * f * t) for f in freqs)
        voltage = 1.0 + 0.002 * t + sum(0.02 * np.cos(2 * np.pi * f * t + 0.4) for f in freqs)
        m = _mfa(freqs, fs, n, voltage, current)
        found = m.search_freq_indexes(m.ft_current)
        self.assertEqual([k - m.index_f0 for k in found], [1, 3, 6])

    def test_closely_spaced_low_frequencies_do_not_crash(self):
        # 0.01/0.03/0.05 Hz in a 1-period (100 s) window: the neighbour spacing is below
        # 4 bins, which used to give an empty search window and a ValueError
        fs, n = 10.0, 1000
        t = np.arange(n) / fs
        freqs = [0.01, 0.03, 0.05]
        sig = sum(np.cos(2 * np.pi * f * t) for f in freqs)
        m = _mfa(freqs, fs, n, sig, sig)
        found = m.search_freq_indexes(m.ft_current)
        self.assertEqual([k - m.index_f0 for k in found], [1, 3, 5])

    def test_failure_leaves_freq_indexes_untouched(self):
        # a frequency beyond Nyquist cannot be found; the failed call must not leave the
        # rough guess behind (a non-None freq_indexes makes callers skip the search forever)
        fs, n = 100.0, 200
        sig = np.random.default_rng(0).normal(size=n)
        m = _mfa([0.5, 80.0], fs, n, sig, sig)
        self.assertIsNone(m.freq_indexes)
        with self.assertRaises(Exception):
            m.search_freq_indexes(m.ft_current)
        self.assertIsNone(m.freq_indexes)

    def test_single_frequency(self):
        fs, n = 100.0, 400
        t = np.arange(n) / fs
        sig = np.cos(2 * np.pi * 2.0 * t)
        m = _mfa([2.0], fs, n, sig, sig)
        self.assertEqual(m.search_freq_indexes(m.ft_current), [m.index_f0 + 8])


if __name__ == "__main__":
    unittest.main()
