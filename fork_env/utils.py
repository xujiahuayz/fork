from typing import Any, Iterable
import numpy as np
from scipy.stats import lognorm, lomax, expon, rv_continuous
from fork_env.constants import (
    HASH_STD,
    SUM_HASH_RATE,
    N_MINER,
)
from scipy.special import erf, gamma, expn, gammaincc, gammainccinv


# create a new distribution class
class truncpl(rv_continuous):
    # https://en.wikipedia.org/wiki/Power_law#Power_law_with_exponential_cutoff

    def __init__(self, alpha: float, ell: float, *args, **kwargs):
        super().__init__(a=0, b=np.inf, *args, **kwargs)
        self.alpha = alpha
        self.ell = ell
        self.one_minus_alpha = 1 - alpha
        # self.scaling_c = ell**one_minus_alpha / gamma(one_minus_alpha)

    def _pdf(self, x: float) -> float:
        return (
            self.ell**self.one_minus_alpha
            * x ** (-self.alpha)
            * np.exp(-self.ell * x)
            / gamma(self.one_minus_alpha)
        )

    def _sf(self, x: float) -> float:
        return gammaincc(1 - self.alpha, self.ell * x)

    def _cdf(self, x: float) -> float:
        return 1 - self._sf(x)

    def _ppf(self, q: float) -> float:
        return gammainccinv(self.one_minus_alpha, 1 - q) / self.ell

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


def calc_truncpl_params(hash_mean: float, hash_std: float) -> tuple[float, float]:
    # parameter fitting using MoM - check mathematica

    one_minus_alpha = (hash_mean / hash_std) ** 2
    alpha = 1 - one_minus_alpha
    # ell = gamma(2 - alpha) / gamma(1 - alpha) / hash_mean
    ell = hash_mean / hash_std**2

    # scaling_c = ell**one_minus_alpha / gamma(one_minus_alpha)
    return alpha, ell


def gen_truncpl_dist(hash_mean: float, hash_std: float) -> tuple[float, float, Any]:
    alpha, ell = calc_truncpl_params(hash_mean, hash_std)
    return (alpha, ell, truncpl(alpha=alpha, ell=ell))


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
    _, _, tp_dist = gen_truncpl_dist(hash_mean=hash_mean, hash_std=hash_std)
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


# if name is main
if __name__ == "__main__":
    # test truncpl
    hash_mean = 1
    hash_std = 0.1
    alpha, ell = calc_truncpl_params(hash_mean, hash_std)
    print(alpha, ell)
    alpha, ell, tp_dist = gen_truncpl_dist(hash_mean, hash_std)
    print(tp_dist.pdf(1))
    print(tp_dist.cdf(1))
    print(tp_dist.sf(1))
    tp_dist.rvs(size=20)
    print(tp_dist.stats(moments="mvsk"))
    tp_dist.cdf(1.04992708)

    tp_dist._ppf(0.7)

    # # test lognorm
    # lognorm_loc, lognorm_sigma, lognorm_scale = calc_ln_params(hash_mean, hash_std)
    # print(lognorm_loc, lognorm_sigma, lognorm_scale)
    # lognorm_loc, lognorm_sigma, lognorm_scale, ln_dist = gen_ln_dist(
    #     hash_mean, hash_std
    # )
    # print(ln_dist.pdf(1))
    # print(ln_dist.cdf(1))
    # print(ln_dist.sf(1))
    # print(ln_dist.rvs(size=10))
    # print(ln_dist.stats(moments="mvsk"))

    # # test lomax
    # lmx_shape, lmx_scale = calc_lmx_params(hash_mean, hash_std)
    # print(lmx_shape, lmx_scale)
    # lmx_shape, lmx_scale, lmx_dist = gen_lmx_dist(hash_mean, hash_std)
    # print(lmx_dist.pdf(1))
    # print(lmx_dist.cdf(1))
    # print(lmx_dist.sf(1))
    # print(lmx_dist.rvs(size=10))
    # print(lmx_dist.stats(moments="mvsk"))

    # # test expon
    # expon_dist = expon_dist(hash_mean)
    # print(expon_dist.pdf(1))
    # print(expon_dist.cdf(1))
    # print(expon_dist.sf(1))
    # print(expon_dist.rvs(size=10))
    # print(expon_dist.stats(moments="mvsk"))

    # # test ccdf_p
    # lbda = 1
    # bis = [1, 2, 3]
    # factor = 1
    # print(ccdf_p(lbda, bis, factor))
    # print(ccdf_p(lbda, bis, factor))
    # print(ccdf_p(lbda, bis, factor))
    # print(ccdf_p
