import unittest
import warnings

import numpy as np

from deistools.processing import FermiDiracFilter


def _original(fr, fc, bw, n):
    X = (fr - fc) / bw
    return np.cosh(n) / (np.cosh(n * X) + np.cosh(n))


class FermiDirac(unittest.TestCase):
    def test_matches_the_cosh_formula_where_that_is_finite(self):
        f = np.linspace(-5, 5, 2001)
        for n in (1, 2, 8, 25):
            np.testing.assert_allclose(FermiDiracFilter(f, 0.3, 1.2, n).values, _original(f, 0.3, 1.2, n),
                                       rtol=1e-10, atol=1e-300)

    def test_no_overflow_warning_for_steep_filters(self):
        f = np.linspace(-500, 500, 4001)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y = FermiDiracFilter(f, 0, 1.0, 100).values
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreater(y[2000], 0.99)         # passband
        self.assertLess(y[0], 1e-100)             # far stopband underflows to ~0, not nan/inf


if __name__ == "__main__":
    unittest.main()
