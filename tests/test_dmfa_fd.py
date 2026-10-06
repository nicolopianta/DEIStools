import unittest

import numpy as np
from numpy.fft import fft, fftfreq, fftshift, ifft, ifftshift

from deistools.processing import FermiDiracFilter, MultiFrequencyAnalysis
from deistools.processing.dmfa_functions import block_average, extract_impedance_fd, extract_zero_frequency_fd


def _reference_dmfa(ft_v, ft_i, freq_axis, indexes, bw_zero, zero_order, bw_filter, order, npts, dt):
    """The straightforward full-array implementation (all frequencies kept at full length,
    block-averaged at the end) that run_dmfa_fd must reproduce with less memory."""
    n = freq_axis.size
    flt0 = FermiDiracFilter(freq_axis, 0, bw_zero, zero_order)
    v0 = n * ifft(ifftshift(ft_v * flt0.values)).real
    i0 = n * ifft(ifftshift(ft_i * flt0.values)).real
    z = np.zeros((len(indexes), n), dtype="complex128")
    for k, f in enumerate(indexes):
        flt = FermiDiracFilter(freq_axis, freq_axis[f], bw_filter, order)
        z[k] = ifft(ifftshift(ft_v * flt.values)) / ifft(ifftshift(ft_i * flt.values))
    block = n // npts
    return (block_average(z, block), block_average(v0, block), block_average(i0, block),
            block_average(dt * np.arange(n), block))


class RunDmfaFd(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.fs, self.n, self.npts = 50.0, 20000, 20
        t = np.arange(self.n) / self.fs
        self.freqs = np.array([0.05, 0.15, 0.4])
        i = 1e-4 + sum(1e-5 * np.cos(2 * np.pi * f * t + p) for f, p in zip(self.freqs, (0.3, 1.1, 2.0)))
        v = 0.5 + 1e-4 * t / t[-1] + sum(2e-4 * np.cos(2 * np.pi * f * t + p) for f, p in zip(self.freqs, (0.5, 0.2, 1.4)))
        self.v = v + rng.normal(0, 1e-5, self.n)
        self.i = i + rng.normal(0, 1e-7, self.n)
        self.mfa = MultiFrequencyAnalysis(self.freqs, self.v, self.i, 1 / self.fs)
        self.mfa.compute_freq_axis()
        self.mfa.compute_fft()
        self.mfa.search_freq_indexes(self.mfa.ft_current)

    def test_matches_reference_implementation(self):
        z, v0, i0, t = self.mfa.run_dmfa_fd(
            time_resolution=1 / self.fs, bw_zero=0.04, bw_filter=0.005, filter_order=8,
            Npts_elab=self.npts, zero_filter_order=100)
        zr, v0r, i0r, tr = _reference_dmfa(
            self.mfa.ft_voltage, self.mfa.ft_current, self.mfa.freq_axis, self.mfa.freq_indexes,
            0.04, 100, 0.005, 8, self.npts, 1 / self.fs)
        self.assertEqual(z.shape, (3, self.npts))
        np.testing.assert_allclose(z, zr, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v0, v0r, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(i0, i0r, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(t, tr)

    def test_recovers_the_impedance_and_the_dc_part(self):
        z, v0, i0, _ = self.mfa.run_dmfa_fd(
            time_resolution=1 / self.fs, bw_zero=0.04, bw_filter=0.005, filter_order=8,
            Npts_elab=self.npts)
        # tone Z = (2e-4 / 1e-5) * exp(1j * (phase_v - phase_i)) = 20 ohm at phases 0.2, -0.9, -0.6
        mid = slice(5, 15)
        np.testing.assert_allclose(np.abs(z[:, mid]), 20.0, rtol=0.03)
        np.testing.assert_allclose(np.angle(z[:, mid]).mean(axis=1), [0.2, -0.9, -0.6], atol=0.05)
        np.testing.assert_allclose(i0[mid], 1e-4, rtol=0.02)
        np.testing.assert_allclose(v0[mid], 0.5, atol=1e-3)  # the DC potential

    def test_old_run_dmfa_signature_untouched(self):
        import inspect
        params = list(inspect.signature(MultiFrequencyAnalysis.run_dmfa).parameters)
        self.assertEqual(params, ["self", "filter", "time_resolution", "Npts_elab"])


class ExtractFunctions(unittest.TestCase):
    def test_block_average(self):
        x = np.arange(12.0)
        np.testing.assert_allclose(block_average(x, 4), [1.5, 5.5, 9.5])
        np.testing.assert_allclose(block_average(np.vstack([x, 2 * x]), 6), [[2.5, 8.5], [5.0, 17.0]])


if __name__ == "__main__":
    unittest.main()
