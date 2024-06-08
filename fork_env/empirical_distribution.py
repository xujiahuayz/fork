from functools import lru_cache
import numpy as np

from typing import Iterable
from scipy.integrate import quad


# E^(b^2/(2*l) + l/2 - b*Cosh[t])*Cosh[t]


# @lru_cache(maxsize=None)
def k_integrand(t: float, bi: float, lbda: float) -> float:
    return np.exp(bi**2 / (2 * lbda) + lbda / 2 - bi * np.cosh(t)) * np.cosh(t)


def density_component(lbda: float, bi: float) -> float:
    if bi == 0:
        return np.exp(lbda / 2)
    if lbda == 0:
        return np.inf
    return (
        bi
        * quad(lambda t: k_integrand(t, bi, lbda), 0, np.inf, epsrel=1e-3, epsabs=1e-4)[
            0
        ]
    )


# modified bessel function, first function second kind
def density_p(lbda: float, bis: Iterable[float]) -> float:
    return np.mean(
        [1 / (2 * density_component(lbda=lbda, bi=bi)) for bi in bis]
    )  # type: ignore


def cdf_p(lbda: float, bis: Iterable[float]) -> float:
    return quad(lambda x: density_p(x, bis), 0, lbda)[0]


def ccdf_p(lbda: float, bis: Iterable[float]) -> float:
    return 1 - cdf_p(lbda, bis)


if "__main__" == __name__:
    print(cdf_p(60, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 9, 40, 50, 900, 1000]))
