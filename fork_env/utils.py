from typing import Any
import numpy as np
from scipy.stats import lognorm, lomax


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
