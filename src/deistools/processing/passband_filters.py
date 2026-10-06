import numpy as np
from dataclasses import dataclass, field

def create_fermi_dirac_fun(fr, fc, bw, n):
    """
    Create a digital filter shaped as a symmetric Fermi-Dirac function.
    """
    X = (fr - fc) / bw
    # Y = cosh(n) / (cosh(n*X) + cosh(n)), written with exponentials of
    # non-positive-or-clipped arguments so that large orders or far-away
    # frequencies underflow to 0 instead of overflowing cosh() to inf.
    a = n * np.abs(X)
    ratio = np.exp(np.minimum(a - n, 700.0)) * (1 + np.exp(-2 * a)) / (1 + np.exp(-2.0 * n))
    return 1.0 / (1.0 + ratio)

@dataclass
class FermiDiracFilter:
    frequencies : np.array
    centered_frequency : float
    cutoff: int
    order : int
    values : np.array = field(init=False) 

    def __post_init__(self):
        self.values = create_fermi_dirac_fun(
            self.frequencies,
            self.centered_frequency,
            self.cutoff,
            self.order
        )
