import unittest

import numpy as np

from deistools.processing import smoothing as S

try:
    from scipy import ndimage, signal
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def noisy(n=200, seed=0, noise=0.2):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6, n)
    return t, np.sin(t) + 0.3 * t + noise * rng.standard_normal(n)


class Helpers(unittest.TestCase):
    def test_odd_window_and_window_points(self):
        self.assertEqual([S.odd_window(p) for p in (1, 2, 4, 5, 6.4)], [3, 3, 5, 5, 7])
        self.assertEqual(S.window_points(seconds=100.0, dt=20.0), 5)
        self.assertEqual(S.window_points(seconds=100.0, dt=20.0, minimum=7), 7)

    def test_segment_bounds(self):
        np.testing.assert_array_equal(S.segment_bounds([4, 6, 5]), [4, 10])
        self.assertEqual(S.segment_bounds([7]).size, 0)

    def test_savgol_coefficients(self):
        np.testing.assert_allclose(S.savgol_coefficients(5, 2), np.array([-3, 12, 17, 12, -3]) / 35)
        np.testing.assert_allclose(S.savgol_coefficients(5, 2, deriv=1), np.array([-2, -1, 0, 1, 2]) / 10, atol=1e-12)
        with self.assertRaises(ValueError):
            S.savgol_coefficients(4, 2)
        with self.assertRaises(ValueError):
            S.savgol_coefficients(3, 3)


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class AgainstScipy(unittest.TestCase):
    def test_savgol_all_edge_modes(self):
        _, y = noisy()
        for window, order in ((5, 2), (11, 3), (21, 2)):
            for edge, scipy_mode in ((S.EDGE_INTERP, "interp"), (S.EDGE_NEAREST, "nearest"), (S.EDGE_MIRROR, "mirror")):
                got = S.smooth(y, S.METHOD_SAVGOL, window=window, polyorder=order, edge=edge)
                want = signal.savgol_filter(y, window, order, mode=scipy_mode)
                np.testing.assert_allclose(got, want, atol=1e-10, err_msg=f"{window} {order} {edge}")

    def test_savgol_derivative(self):
        t, y = noisy(noise=0.0)
        got = S.smooth(y, S.METHOD_SAVGOL, window=9, polyorder=3, deriv=1, delta=t[1] - t[0], edge=S.EDGE_INTERP)
        want = signal.savgol_filter(y, 9, 3, deriv=1, delta=t[1] - t[0], mode="interp")
        np.testing.assert_allclose(got, want, atol=1e-8)

    def test_gaussian(self):
        _, y = noisy()
        for sigma in (0.7, 2.0, 5.0):
            for edge, scipy_mode in ((S.EDGE_NEAREST, "nearest"), (S.EDGE_MIRROR, "mirror")):
                got = S.smooth(y, S.METHOD_GAUSSIAN, sigma=sigma, edge=edge)
                want = ndimage.gaussian_filter1d(y, sigma, mode=scipy_mode, truncate=4.0)
                np.testing.assert_allclose(got, want, atol=1e-12)

    def test_moving_average_and_median(self):
        _, y = noisy()
        np.testing.assert_allclose(S.smooth(y, S.METHOD_MEAN, window=7),
                                   ndimage.uniform_filter1d(y, 7, mode="nearest"), atol=1e-12)
        np.testing.assert_allclose(S.smooth(y, S.METHOD_MEDIAN, window=7),
                                   ndimage.median_filter(y, size=7, mode="nearest"))


class Properties(unittest.TestCase):
    def test_a_constant_is_preserved_by_every_method(self):
        y = np.full(50, 3.5)
        for method in S.METHODS:
            for edge in S.EDGE_MODES:
                np.testing.assert_allclose(S.smooth(y, method, window=9, sigma=2, edge=edge), y, atol=1e-12)

    def test_savgol_keeps_a_polynomial_exactly_including_the_edges(self):
        x = np.linspace(-1, 1, 60)
        y = 2 - 3 * x + 0.5 * x ** 2 + 4 * x ** 3
        got = S.smooth(y, S.METHOD_SAVGOL, window=11, polyorder=3, edge=S.EDGE_INTERP)
        np.testing.assert_allclose(got, y, atol=1e-10)

    def test_smoothing_reduces_the_noise(self):
        t, y = noisy(n=600, noise=0.3)
        clean = np.sin(t) + 0.3 * t
        for method in (S.METHOD_SAVGOL, S.METHOD_GAUSSIAN, S.METHOD_MEAN, S.METHOD_MEDIAN):
            smoothed = S.smooth(y, method, window=15, sigma=4, polyorder=3)
            self.assertLess(np.std(smoothed - clean), 0.5 * np.std(y - clean), method)

    def test_none_returns_a_copy(self):
        y = np.arange(5.0)
        out = S.smooth(y, S.METHOD_NONE)
        np.testing.assert_array_equal(out, y)
        out[0] = 99
        self.assertEqual(y[0], 0)

    def test_segments_are_filtered_independently(self):
        y = np.concatenate([np.full(20, 1.0), np.full(20, 5.0)])
        blurred = S.smooth(y, S.METHOD_GAUSSIAN, sigma=3)
        self.assertGreater(blurred[19], 1.5)                       # without segments the step is blurred
        kept = S.smooth(y, S.METHOD_GAUSSIAN, sigma=3, segments=[20])
        np.testing.assert_allclose(kept, y, atol=1e-12)            # with them it stays a step
        medians = S.smooth(y + np.arange(40) * 0.0, S.METHOD_SAVGOL, window=9, polyorder=2, segments=S.segment_bounds([20, 20]))
        np.testing.assert_allclose(medians, y, atol=1e-10)

    def test_a_short_segment_gets_a_smaller_window_or_is_left_alone(self):
        y = np.concatenate([np.arange(30.0), [100.0, 200.0], np.arange(30.0)])
        out = S.smooth(y, S.METHOD_MEAN, window=9, segments=[30, 32])
        np.testing.assert_array_equal(out[30:32], [100.0, 200.0])  # 2 points: unchanged
        self.assertEqual(out.shape, y.shape)
        short = S.smooth(np.array([1.0, 5.0, 2.0, 8.0, 3.0]), S.METHOD_SAVGOL, window=21, polyorder=2)
        self.assertEqual(short.shape, (5,))                        # window shrunk, no error
        self.assertTrue(np.all(np.isfinite(short)))

    def test_nan_gaps_stay_gaps_and_do_not_spread(self):
        _, y = noisy()
        y[50:53] = np.nan
        out = S.smooth(y, S.METHOD_GAUSSIAN, sigma=2)
        self.assertTrue(np.all(np.isnan(out[50:53])))
        self.assertEqual(int(np.isnan(out).sum()), 3)
        self.assertTrue(np.all(np.isnan(S.smooth(np.full(10, np.nan), S.METHOD_MEAN, window=3))))

    def test_complex_data_are_filtered_as_real_and_imaginary_parts(self):
        _, a = noisy(seed=1)
        _, b = noisy(seed=2)
        z = a + 1j * b
        out = S.smooth(z, S.METHOD_SAVGOL, window=9, polyorder=2)
        np.testing.assert_allclose(out.real, S.smooth(a, S.METHOD_SAVGOL, window=9, polyorder=2))
        np.testing.assert_allclose(out.imag, S.smooth(b, S.METHOD_SAVGOL, window=9, polyorder=2))

    def test_any_axis_and_shape(self):
        rng = np.random.default_rng(3)
        z = rng.standard_normal((6, 80)) + 1j * rng.standard_normal((6, 80))
        along_time = S.smooth(z, S.METHOD_GAUSSIAN, sigma=2, axis=1)
        row = S.smooth(z[2], S.METHOD_GAUSSIAN, sigma=2)
        np.testing.assert_allclose(along_time[2], row)
        np.testing.assert_allclose(S.smooth(z.T, S.METHOD_GAUSSIAN, sigma=2, axis=0), along_time.T)

    def test_input_is_not_modified_and_bad_arguments_are_refused(self):
        y = np.arange(10.0)
        before = y.copy()
        S.smooth(y, S.METHOD_MEDIAN, window=5)
        np.testing.assert_array_equal(y, before)
        with self.assertRaises(ValueError):
            S.smooth(y, "Kalman")
        with self.assertRaises(ValueError):
            S.smooth(y, S.METHOD_MEAN, edge="wrap")
        with self.assertRaises(ValueError):
            S.smooth(y, S.METHOD_SAVGOL, window=5, polyorder=5)


class Impedance(unittest.TestCase):
    def test_polar_handles_a_phase_that_wraps(self):
        t = np.linspace(0, 1, 120)
        phase = 6.0 * t - 3.0                   # runs from -3 to +3 rad then continues past pi? keep going
        z = 10.0 * np.exp(1j * (phase + 3.0))   # crosses +-pi once
        wrapped = np.angle(z)
        self.assertGreater(np.abs(np.diff(wrapped)).max(), 3.0)       # there is a jump in the raw angle
        out = S.smooth_impedance(z, S.METHOD_SAVGOL, representation=S.REPRESENTATION_POLAR, window=9, polyorder=2,
                                 edge=S.EDGE_INTERP)
        np.testing.assert_allclose(np.abs(out), 10.0, atol=1e-9)
        np.testing.assert_allclose(out, z, atol=1e-8)                 # a clean arc is returned unchanged
        cartesian = S.smooth_impedance(z, S.METHOD_GAUSSIAN, sigma=4)
        self.assertLess(np.abs(cartesian[60]), 10.0)                  # the chord of the arc: smaller than the radius

    def test_polar_on_noisy_data_beats_the_raw_series(self):
        rng = np.random.default_rng(4)
        truth = 50.0 * np.exp(-1j * 0.4) * np.ones(300)
        z = truth * (1 + 0.05 * rng.standard_normal(300)) * np.exp(1j * 0.05 * rng.standard_normal(300))
        out = S.smooth_impedance(z, S.METHOD_GAUSSIAN, representation=S.REPRESENTATION_POLAR, sigma=5)
        self.assertLess(np.std(np.abs(out - truth)), 0.5 * np.std(np.abs(z - truth)))

    def test_polar_segments_and_rows(self):
        z = np.vstack([np.concatenate([np.full(15, 5 * np.exp(-0.2j)), np.full(15, 9 * np.exp(-0.9j))]),
                       np.full(30, 7 + 0j)])
        out = S.smooth_impedance(z, S.METHOD_GAUSSIAN, representation=S.REPRESENTATION_POLAR, sigma=3,
                                 segments=[15], axis=1)
        np.testing.assert_allclose(out, z, atol=1e-10)

    def test_unknown_representation_is_refused(self):
        with self.assertRaises(ValueError):
            S.smooth_impedance(np.ones(5, dtype=complex), S.METHOD_MEAN, representation="log")


class Hampel(unittest.TestCase):
    def test_a_spike_is_replaced_by_the_median(self):
        _, y = noisy(noise=0.05)
        spiked = y.copy()
        spiked[80] += 5.0
        cleaned, mask = S.hampel(spiked, window=15, n_sigmas=4.0)
        self.assertTrue(mask[80])
        self.assertLess(abs(cleaned[80] - y[80]), 0.3)
        self.assertLess(mask.sum(), 0.05 * y.size)              # almost nothing else is touched
        np.testing.assert_array_equal(cleaned[~mask], spiked[~mask])

    def test_a_spike_on_a_flat_series_is_found_and_flat_noise_is_not(self):
        y = np.full(40, 2.0)
        y[10] = 9.0
        cleaned, mask = S.hampel(y, window=7)
        self.assertEqual(np.flatnonzero(mask).tolist(), [10])
        np.testing.assert_allclose(cleaned, 2.0)
        rounding = np.full(40, 2.0) + 1e-15 * np.arange(40)
        self.assertFalse(S.hampel(rounding)[1].any())

    def test_complex_points_are_replaced_as_a_whole(self):
        rng = np.random.default_rng(5)
        z = (20 + 3j) + 0.05 * (rng.standard_normal(60) + 1j * rng.standard_normal(60))
        z[30] = 35 - 8j
        cleaned, mask = S.hampel(z, window=9)
        self.assertTrue(mask[30])
        self.assertLess(abs(cleaned[30] - (20 + 3j)), 0.5)

    def test_segments_and_nans(self):
        y = np.concatenate([np.full(20, 1.0), np.full(20, 8.0)])
        _, mask = S.hampel(y, window=7, segments=[20])
        self.assertFalse(mask.any())                           # the step is a level change, not an outlier
        y2 = np.full(30, 4.0)
        y2[5] = np.nan
        cleaned, mask = S.hampel(y2, window=7)
        self.assertTrue(np.isnan(cleaned[5]) and not mask[5])


if __name__ == "__main__":
    unittest.main()
