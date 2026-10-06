import unittest

import numpy as np

from deistools.processing import MultiFrequencyAnalysis


class Bltva(unittest.TestCase):
    def setUp(self):
        fs, T = 100.0, 20.0
        n = int(fs * T)
        t = np.arange(n) / fs
        self.T = T
        self.freqs = np.array([1.0, 2.0, 4.0, 8.0])
        self.z = np.array([10 + 0j, 8 - 2j, 6 - 3j, 5 - 1j])
        phases = np.array([0.3, 1.0, 2.0, 0.7])
        self.i = sum(1e-3 * np.cos(2 * np.pi * f * t + p) for f, p in zip(self.freqs, phases))
        self.v = 0.2 * t / T + sum(1e-3 * abs(z) * np.cos(2 * np.pi * f * t + p + np.angle(z))
                                   for f, p, z in zip(self.freqs, phases, self.z))
        self.mfa = MultiFrequencyAnalysis(self.freqs, self.v, self.i, 1 / fs)

    def test_recovers_impedance_despite_drift_and_exposes_diagnostics(self):
        z_t, drift = self.mfa.run_bltva_eis(T_meas=self.T, Np=1, Nq=1, Na=1)
        self.assertEqual(z_t.shape, (4, self.v.size))
        self.assertEqual(drift.shape, z_t.shape)
        np.testing.assert_allclose(np.median(z_t, axis=1), self.z, atol=1e-6)
        diag = self.mfa.bltva_diagnostics
        self.assertEqual(len(diag), 4)
        for key in ("V_fit", "V_meas", "k_band_bins", "Z_p", "theta_q", "cond_K", "residual_rel"):
            self.assertIn(key, diag[0])
        self.assertEqual(diag[0]["V_fit"].shape, diag[0]["V_meas"].shape)

    def test_polynomial_orders_are_arguments(self):
        z_t, _ = self.mfa.run_bltva_eis(T_meas=self.T, Np=2, Nq=2, Na=2)
        np.testing.assert_allclose(np.median(z_t, axis=1), self.z, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
