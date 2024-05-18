import numpy as np


def calc_ex_rate(hash_mean: float) -> float:
    return 1 / hash_mean


def calc_ln_sig(hash_mean: float, hash_std: float) -> float:
    return np.sqrt(np.log(1 + hash_std**2 / hash_mean**2))


def calc_ln_loc(hash_mean: float, hash_std: float) -> float:
    lognorm_sigma = calc_ln_sig(hash_mean, hash_std)
    return np.log(hash_mean) - lognorm_sigma**2 / 2


def calc_ln_scale(hash_mean: float, hash_std: float) -> float:
    lognorm_loc = calc_ln_loc(hash_mean, hash_std)
    return np.exp(lognorm_loc)


def calc_lmx_shape(hash_mean: float, hash_std: float) -> float:
    return 2 * hash_std**2 / (hash_std**2 - hash_mean**2)


def calc_lmx_scale(hash_mean: float, hash_std: float) -> float:
    lomax_shape = calc_lmx_shape(hash_mean, hash_std)
    return hash_mean * (lomax_shape - 1)
