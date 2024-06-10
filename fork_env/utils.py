from math import prod
from typing import Any, Iterable
import numpy as np
from scipy.stats import lognorm, lomax
from scipy.integrate import quad
from scipy.special.cython_special import kn
from scipy.stats import rv_continuous


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


def density_component(lbda: float, bi: float) -> float:
    if bi == 0:
        return np.exp(lbda / 2)
    if lbda == 0:
        return np.inf
    return bi * np.exp(bi**2 / (2 * lbda) + lbda / 2) * kn(1, bi)


# modified bessel function, first function second kind
def density_p(lbda: float, bis: Iterable[float]) -> float:
    return np.mean(
        [1 / (2 * density_component(lbda=lbda, bi=bi)) for bi in bis]
    )  # type: ignore


def cdf_p(lbda: float, bis: Iterable[float]) -> float:
    return quad(lambda x: density_p(x, bis), 0, lbda)[0]


def ccdf_p(lbda: float, bis: Iterable[float]) -> float:
    return 1 - cdf_p(lbda, bis)


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

    def _pdf(self, x: float) -> float:
        return density_p(
            x,
            bis=self.bis,
        )


if __name__ == "__main__":
    emp_dist = EmpDist(
        bis=[
            0.0004971774441377959,
            0.0004360584338864814,
            0.0001990651875942347,
            0.0001844994421605102,
            0.00011464098464896097,
            7.562763511471067e-05,
            6.517457074462605e-05,
            1.7764497372111044e-05,
            1.747889452046939e-05,
            1.685056824685774e-05,
            1.5136951137007803e-05,
            1.4794227715037815e-05,
            1.4622866004052821e-05,
            1.376605744912785e-05,
            9.653376385487993e-06,
            7.825518134981392e-06,
            3.198751938386554e-06,
            1.599375969193277e-06,
            1.4851348285366145e-06,
            1.370893687879952e-06,
            1.3137731175516207e-06,
            1.0281702659099639e-06,
            6.85446843939976e-07,
            6.283262736116446e-07,
            3.998439922983193e-07,
            3.42723421969988e-07,
            2.2848228131332533e-07,
            2.2848228131332533e-07,
            1.71361710984994e-07,
            1.71361710984994e-07,
            1.1424114065666266e-07,
            1.1424114065666266e-07,
            1.1424114065666266e-07,
            1.1424114065666266e-07,
            5.712057032833133e-08,
            5.712057032833133e-08,
            5.712057032833133e-08,
            5.712057032833133e-08,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    print(emp_dist.rvs(size=2))
