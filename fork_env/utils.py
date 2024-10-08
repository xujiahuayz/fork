from typing import Any, Iterable
import numpy as np
from scipy.stats import lognorm, lomax, expon, rv_continuous
from fork_env.constants import (
    HASH_STD,
    SUM_HASH_RATE,
    N_MINER,
)
from scipy.special import erf, gamma, expn


# create a new distribution class
class truncpl(rv_continuous):
    # https://en.wikipedia.org/wiki/Power_law#Power_law_with_exponential_cutoff

    def __init__(self, alpha: float, ell: float, scaling_c: float, *args, **kwargs):
        super().__init__(a=0, b=np.inf, *args, **kwargs)
        self.alpha = alpha
        self.ell = ell
        self.scaling_c = scaling_c

    def _pdf(self, x: float) -> float:
        return self.scaling_c * x ** (-self.alpha) * np.exp(-self.ell * x)

    # def pdf(self, x: float | Interable[float]) -> float | Iterable[float]:

    # def _sf(self, x):
    #     return (
    #         self.scaling_c
    #         * x ** (-self.alpha)
    #         * (
    #             x * expn(self.alpha, self.ell * x)
    #             + (self.ell * x) ** self.alpha * gamma(1 - self.alpha) / self.ell
    #         )
    #     )


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


def calc_truncpl_params(
    hash_mean: float, hash_std: float
) -> tuple[float, float, float]:
    # parameter fitting using MoM - check mathematica

    alpha = 1 - (hash_mean / hash_std) ** 2
    ell = gamma(2 - alpha) / gamma(1 - alpha) / hash_mean

    scaling_c = ell ** (1 - alpha) / gamma(1 - alpha)
    return alpha, ell, scaling_c


def gen_truncpl_dist(
    hash_mean: float, hash_std: float
) -> tuple[float, float, float, Any]:
    alpha, ell, scaling_c = calc_truncpl_params(hash_mean, hash_std)
    return (alpha, ell, scaling_c, truncpl(alpha=alpha, ell=ell, scaling_c=scaling_c))


def expon_dist(hash_mean: float):
    return expon(scale=hash_mean)


def lognorm_dist(
    hash_mean: float,
    hash_std: float = HASH_STD,
):
    _, _, ln_dist = gen_ln_dist(hash_mean=hash_mean, hash_std=hash_std)
    return ln_dist


def lomax_dist(hash_mean: float, hash_std: float = HASH_STD):
    _, _, lmx_dist = gen_lmx_dist(hash_mean=hash_mean, hash_std=hash_std)
    return lmx_dist


def truncpl_dist(hash_mean: float, hash_std: float = HASH_STD):
    _, _, _, tp_dist = gen_truncpl_dist(hash_mean=hash_mean, hash_std=hash_std)
    return tp_dist


def ccdf_p(lbda: float, bis: list[int], factor: float) -> float:
    fl = factor * lbda
    bis = np.array(bis)
    return (
        np.mean(
            1
            + erf((bis - fl) / np.sqrt(2 * fl))
            + np.exp(2 * bis + np.log(1 - erf((bis + fl) / np.sqrt(2 * fl))))
        )
        / 2
    )
