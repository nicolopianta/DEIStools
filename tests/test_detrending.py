import unittest

import numpy as np

from deistools.processing import detrending as dt


def _record(fs=10.0, n_blocks=40, f_min=0.1):
    block = int(round(fs / f_min))
    n = n_blocks * block
    t = np.arange(n) / fs
    freqs = np.array([0.1, 0.2, 0.5])  # integer multiples of f_min
    tones = sum(0.01 * np.cos(2 * np.pi * f * t + p) for f, p in zip(freqs, (0.3, 1.2, 2.2)))
    return fs, t, freqs, tones


class Detrend(unittest.TestCase):
    def test_none_is_identity(self):
        fs, t, freqs, tones = _record()
        x, trend = dt.detrend(tones + 1.0, fs, freqs, dt.METHOD_NONE)
        np.testing.assert_array_equal(x, tones + 1.0)
        self.assertFalse(trend.any())

    def test_block_means_and_end_point_line_are_blind_to_tones(self):
        fs, t, freqs, tones = _record()
        x = 2.0 + 0.01 * t + tones
        detrended, trend = dt.detrend(x, fs, freqs, dt.METHOD_LINE)
        np.testing.assert_allclose(x, detrended + trend)
        # a whole number of periods of every tone in each end block: the line is exact
        np.testing.assert_allclose(trend, 2.0 + 0.01 * t, atol=1e-9)
        np.testing.assert_allclose(detrended, tones, atol=1e-9)

    def test_spline_follows_curved_drift_without_eating_the_tones(self):
        fs, t, freqs, tones = _record()
        drift = 1.0 + 2.0 * np.exp(-t / 120.0) + 0.002 * t
        detrended, trend = dt.detrend(drift + tones, fs, freqs, dt.METHOD_SPLINE, knot_spacing_periods=3.0)
        inner = slice(int(0.1 * t.size), int(0.9 * t.size))
        np.testing.assert_allclose(trend[inner], drift[inner], atol=2e-3)
        # tone amplitude is preserved: residual between trend-removed signal and tones is small
        self.assertLess(np.sqrt(np.mean((detrended[inner] - tones[inner]) ** 2)), 0.1 * 0.01)

    def test_optional_edge_refinement_follows_a_fast_relaxation_at_the_start(self):
        fs, t, freqs, tones = _record(n_blocks=60)
        drift = 1.0 + 1.5 * np.exp(-t / 40.0) + 0.0005 * t              # time constant 40 s
        x = drift + tones
        _, plain = dt.detrend(x, fs, freqs, dt.METHOD_SPLINE, knot_spacing_periods=3.0)
        _, refined = dt.detrend(x, fs, freqs, dt.METHOD_SPLINE, knot_spacing_periods=3.0, edge_knots=6, fine=10)
        head = slice(0, 100)  # first 10 s
        self.assertLess(np.abs(refined - drift)[head].max(), np.abs(plain - drift)[head].max() / 5)
        self.assertLess(np.abs(refined - drift)[head].max(), 0.05)

    def test_default_spline_knots_are_the_block_centres_every_n_blocks(self):
        # pins the default knot placement: the original formulation took every n-th block centre
        # (cc[n], cc[2n], ...) that lies strictly inside (cc[1], cc[-2])
        from scipy.interpolate import LSQUnivariateSpline
        fs, t, freqs, tones = _record(n_blocks=54)               # 54 blocks: a knot of 10 falls just before the end
        x = 1.0 + 2.0 * np.exp(-t / 120.0) + 0.002 * t + tones
        block = 100
        yc, cc = dt.block_means(x, block)
        for n_blocks in (3, 10):
            step = n_blocks
            knots = cc[step::step]
            knots = knots[(knots > cc[1]) & (knots < cc[-2])]
            reference = LSQUnivariateSpline(cc, yc, knots, k=3)(np.arange(x.size, dtype=float))
            np.testing.assert_allclose(dt.fit_smooth_trend(x, block, dt.METHOD_SPLINE, 3, n_blocks), reference, atol=1e-12)

    def test_polynomial_fits_polynomial_drift(self):
        fs, t, freqs, tones = _record()
        drift = 1.0 + 0.003 * t - 2e-6 * t ** 2
        detrended, trend = dt.detrend(drift + tones, fs, freqs, dt.METHOD_POLY, order=2)
        np.testing.assert_allclose(trend, drift, atol=1e-3)

    def test_too_short_record_is_refused(self):
        fs, t, freqs, tones = _record(n_blocks=3)
        with self.assertRaises(ValueError):
            dt.detrend(tones, fs, freqs, dt.METHOD_SPLINE)

    def test_tone_signal_extracts_only_the_tone_bands(self):
        fs, t, freqs, tones = _record()
        out = dt.tone_signal(tones + 5.0, fs, freqs, half_bw=0.01)
        # the DC offset is outside the bands (and the DC bin is never included)
        self.assertLess(np.sqrt(np.mean((out - tones) ** 2)), 0.3 * np.sqrt(np.mean(tones ** 2)))

    def test_legacy_baseline_functions_still_there(self):
        self.assertTrue(callable(dt.remove_baseline) and callable(dt.redo_baseline))


if __name__ == "__main__":
    unittest.main()
