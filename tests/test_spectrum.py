import unittest

import numpy as np

from deistools.processing.spectrum import amplitude_spectrum, log_binned_max


class Spectrum(unittest.TestCase):
    def test_amplitude_scaling_and_dc_dropped(self):
        fs, n = 50.0, 5000
        t = np.arange(n) / fs
        x = 3.0 + 0.02 * np.cos(2 * np.pi * 0.5 * t)
        f, a = amplitude_spectrum(x, fs)
        self.assertGreater(f[0], 0)
        self.assertAlmostEqual(a[np.argmin(abs(f - 0.5))], 0.02, places=6)

    def test_log_binned_max_keeps_the_peaks_and_the_low_bins(self):
        f = np.arange(1, 200001) * 1e-4
        a = np.full(f.size, 1e-6)
        a[12345] = 0.5
        x, y = log_binned_max(f, a, n_bins=2000)
        self.assertLess(x.size, 2200)
        self.assertEqual(y.max(), 0.5)
        np.testing.assert_array_equal(x[:50], f[:50])  # one bin per point where bins are sparse

    def test_short_spectra_pass_through(self):
        f = np.arange(1, 11.0)
        x, y = log_binned_max(f, f * 2, n_bins=100)
        np.testing.assert_array_equal(x, f)


if __name__ == "__main__":
    unittest.main()
