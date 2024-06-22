from typing import Any
import numpy as np
from scipy.stats import lognorm, lomax
from scipy.stats import rv_continuous
from scipy.integrate import quad_vec
from numba import njit


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


@njit
def ccdf_component(lbda: float, bi: int):
    if bi == 0:
        return 1 / np.exp(lbda / 2)
    return np.exp(bi - pow(bi, 2) / (2 * lbda) - lbda / 2)


def zele(bi: int, B: int) -> float:
    return (
        quad_vec(lambda x: ccdf_component(x, bi), 0, bi)[0]
        + quad_vec(lambda x: ccdf_component(x, bi), bi, B)[0]
    )


def pdf_empirical(lbda: float, bis: list[int], ints: list[float]) -> float:
    return np.mean([ccdf_component(lbda, bis[i]) / w for i, w in enumerate(ints)])  # type: ignore


def ccdf_p(lbda: float, bis: list[int], B: int, ints: list[float]) -> float:
    return np.mean(
        [
            quad_vec(lambda x: ccdf_component(x, bis[i]), lbda, B)[0] / w  # type: ignore
            for i, w in enumerate(ints)
        ]
    )


def cdf_p(lbda: float, bis: list[int], B: int, ints: list[float]) -> float:
    return 1 - ccdf_p(lbda, bis, B, ints)


# class EmpDist(rv_continuous):
#     def __init__(self, bis: list[int], xtol: float = 1e-14, seed=None):
#         super().__init__(a=0, xtol=xtol, seed=seed)
#         self.bis = bis

#     def _cdf(self, x: float) -> float:
#         return cdf_p(x, self.bis)


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
