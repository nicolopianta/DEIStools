import unittest

import numpy as np

from deistools.processing import detrending as dt
from deistools.processing.dmfa_pipeline import run_dmfa_on_record, zero_order_hold_resample


def _synthetic():
    fs, n = 10.0, 20000
    t = np.arange(n) / fs
    freqs = np.array([0.05, 0.1, 0.25])
    w = 2 * np.pi * freqs
    z_true = 1.0 + 3 / (1 + 1j * w * 2.0)
    phases = np.array([0.3, 1.0, 2.0])
    current = -1e-4 + sum(1e-5 * np.cos(2 * np.pi * f * t + p) for f, p in zip(freqs, phases))
    drift = 1.0 + 1.5 * np.exp(-t / 150.0) + 0.0005 * t      # fast start + slow ramp: a ~1 V edge jump
    voltage = drift + sum(1e-5 * abs(z) * np.cos(2 * np.pi * f * t + p + np.angle(z))
                          for f, p, z in zip(freqs, phases, z_true))
    return fs, t, freqs, z_true, voltage, current, drift


class RunDmfaOnRecord(unittest.TestCase):
    def test_detrending_removes_the_edge_leakage_and_keeps_the_dc_level(self):
        fs, t, freqs, z_true, v, i, drift = _synthetic()
        results = {m: run_dmfa_on_record(v, i, freqs, fs, detrend_method=m, knot_spacing_periods=3.0)
                   for m in (dt.METHOD_NONE, dt.METHOD_SPLINE)}
        plain, spline = results[dt.METHOD_NONE], results[dt.METHOD_SPLINE]
        self.assertEqual(spline.impedance.shape, (3, 100))
        self.assertEqual(spline.block_len, 200)
        self.assertEqual(spline.n_samples, 20000)

        def edge_err(r):
            e = np.abs(r.impedance - z_true[:, None]) / np.abs(z_true)[:, None]
            return e[:, :3].max()

        def inner_err(r):
            e = np.abs(r.impedance - z_true[:, None]) / np.abs(z_true)[:, None]
            return np.median(e[:, 10:-10])

        self.assertLess(edge_err(spline), edge_err(plain) / 100)
        self.assertLess(inner_err(spline), 0.01)
        self.assertLess(inner_err(spline), inner_err(plain) / 10)
        # the trend is put back into the DC voltage/current
        dc_v_true = np.interp(spline.time, t, drift)
        np.testing.assert_allclose(spline.zero_voltage[5:-5], dc_v_true[5:-5], atol=3e-3)
        np.testing.assert_allclose(spline.zero_current[5:-5], -1e-4, atol=2e-6)

    def test_no_detrend_matches_plain_dmfa_calls(self):
        fs, t, freqs, z_true, v, i, drift = _synthetic()
        r = run_dmfa_on_record(v, i, freqs, fs)
        self.assertEqual(r.detrend_method, dt.METHOD_NONE)
        self.assertIsNone(r.voltage_trend)
        self.assertEqual(len(r.freq_indexes), 3)

    def test_trims_to_whole_blocks_and_refuses_empty_input(self):
        fs, t, freqs, z_true, v, i, drift = _synthetic()
        r = run_dmfa_on_record(v[:19950], i[:19950], freqs, fs)
        self.assertEqual(r.n_samples, 19800)
        with self.assertRaises(ValueError):
            run_dmfa_on_record(v[:100], i[:100], freqs, fs)


class ZeroOrderHold(unittest.TestCase):
    def test_holds_the_latest_column(self):
        rows = np.array([[1, 2, 3], [10, 20, 30]])
        out = zero_order_hold_resample(rows, np.array([10.0, 20.0, 30.0]), np.array([0.0, 10.0, 15.0, 20.0, 29.9, 99.0]))
        np.testing.assert_array_equal(out, [[1, 1, 1, 2, 2, 3], [10, 10, 10, 20, 20, 30]])


if __name__ == "__main__":
    unittest.main()
