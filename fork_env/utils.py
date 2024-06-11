from math import prod
from typing import Any, Iterable
import numpy as np
from scipy.stats import lognorm, lomax
from scipy.special.cython_special import kn
from scipy.stats import rv_continuous

from mpmath import mp, quad, exp, power

mp.dps = 120  # Set precision to 50 decimal places

# from decimal import Decimal, getcontext

# getcontext().prec = 500
# from sympy import symbols, integrate, sqrt


def calc_ex_rate(hash_mean: float) -> float:
    return 1 / hash_mean


def calc_ln_sig2(hash_mean: float, hash_std: float) -> float:
    return np.log(1 + hash_std**2 / hash_mean**2)


def calc_ln_sig(hash_mean: float, hash_std: float) -> float:
    return np.sqrt(calc_ln_sig2(hash_mean, hash_std))


def calc_ln_params(hash_mean: float, hash_std: float) -> tuple[float, float, float]:
    lognorm_sigma = calc_ln_sig(hash_mean, hash_std)
    lognorm_loc = np.log(hash_mean) - lognorm_sigma**2 / 2  # mu
    lognorm_scale = np.exp(lognorm_loc)
    return lognorm_loc, lognorm_sigma, lognorm_scale


def gen_ln_dist(hash_mean: float, hash_std: float) -> tuple[float, float, Any]:
    lognorm_loc, lognorm_sigma, lognorm_scale = calc_ln_params(hash_mean, hash_std)
    return lognorm_loc, lognorm_sigma, lognorm(lognorm_sigma, scale=lognorm_scale)


def calc_lmx_shape(hash_mean: float, hash_std: float) -> float:
    return 2 * hash_std**2 / (hash_std**2 - hash_mean**2)


def calc_lmx_params(hash_mean: float, hash_std: float) -> tuple[float, float]:
    lmx_shape = calc_lmx_shape(hash_mean, hash_std)
    if lmx_shape <= 2:
        raise ValueError("Lomax shape must be greater than 2 to have a finite std.")
    lmx_scale = hash_mean * (lmx_shape - 1)
    return lmx_shape, lmx_scale


def gen_lmx_dist(hash_mean: float, hash_std: float) -> tuple[float, float, Any]:
    lomax_shape, lmx_scale = calc_lmx_params(hash_mean, hash_std)
    return lomax_shape, lmx_scale, lomax(lomax_shape, scale=lmx_scale)


def ccdf_component(lbda: float, bi: float):
    lbda = mp.mpf(lbda)
    # bi = Decimal(int(bi))
    result = exp(bi - power(bi, 2) / (2 * lbda) - lbda / 2)
    return result


def ccdf_p(lbda: float, bis: list[float]) -> float:
    B = np.sum(bis)
    ints = [float(quad(lambda x: ccdf_component(x, bi), [0, B])) for bi in bis]
    return np.mean(
        [
            float(quad(lambda x: ccdf_component(x, bis[i]), [lbda, B])) / w
            for i, w in enumerate(ints)
        ]
    )


def cdf_p(lbda: float, bis: Iterable[float]) -> float:
    return 1 - ccdf_p(lbda, bis)


def p_delta_emp_integrand(
    x: float, delta: float, bis: Iterable[float], block_window: int
) -> float:
    sum_i = 0
    for i, bi in enumerate(bis):
        sum_j = 0
        for j, bj in enumerate(bis):
            if i != j:
                prod_k = prod(
                    kn(1, bk * np.sqrt(2 * (x + delta) / block_window + 1))
                    / (kn(1, bk) * np.sqrt(2 * (x + delta) / block_window + 1))
                    for k, bk in enumerate(bis)
                    if j != k
                )
                sum_j += (
                    kn(2, bj * np.sqrt(2 * (x + delta) / block_window + 1))
                    / (kn(1, bj) * (2 * (x + delta) + block_window))
                    * prod_k
                )
        sum_i += (
            kn(2, bi * np.sqrt(2 * x / block_window + 1))
            / (kn(1, bi) * (2 * x + block_window))
            * sum_j
        )
    return sum_i


def p_delta_emp(delta: float, bis: Iterable[float], block_window: int) -> float:
    return quad(
        lambda x: p_delta_emp_integrand(x, delta, bis, block_window),
        0,
        np.inf,
    )[0]


def c_delta_emp(delta: float, bis: Iterable[float], block_window: int) -> float:
    return quad(
        lambda d: p_delta_emp(d, bis, block_window),
        0,
        delta,
    )[0]


class EmpDist(rv_continuous):
    def __init__(self, bis: Iterable[float], xtol: float = 1e-14, seed=None):
        super().__init__(a=0, xtol=xtol, seed=seed)
        self.bis = bis

    def _cdf(self, x: float) -> float:
        return cdf_p(x, self.bis)


if __name__ == "__main__":
    # Test the EmpDist class
    bis = [
        8704,
        7633,
        3485,
        3230,
        2007,
        1324,
        1141,
        311,
        306,
        295,
        265,
        259,
        256,
        241,
        169,
        137,
        56,
        28,
        26,
        24,
        23,
        18,
        12,
        11,
        7,
        6,
        4,
        4,
        3,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
    ]
