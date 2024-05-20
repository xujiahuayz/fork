from functools import lru_cache
from typing import Callable, Literal

import numpy as np
from scipy.integrate import dblquad, quad_vec

from fork_env.constants import HASH_STD, SUM_HASH_RATE, EMPIRICAL_PROP_DELAY
from fork_env.utils import (
    calc_lmx_params,
    calc_ln_params,
)


@lru_cache(maxsize=None)
def pdf_log_normal(
    lam: float, mu: float, sigmaSQRRT2PI: float, twosigsq: float
) -> float:
    """
    pdf of the log normal distribution where mean = np.log(sum_lambda / n) - np.square(sigma) / 2 and sigma = sigma
    """
    return np.exp(-((np.log(lam) - mu) ** 2) / twosigsq) / (lam * sigmaSQRRT2PI)


@lru_cache(maxsize=None)
def pdf_lomax(lam: float, lmx_shape: float, lmx_scale: float) -> float:
    """
    pdf of the lomax distribution where scale = sum_lambda * (c - 1) / n and c = c
    """
    return lmx_shape / lmx_scale * (1 + lam / lmx_scale) ** (-lmx_shape - 1)


def az_integrand(
    lam: float,
    x: float,
    pdf: Callable,
) -> float:
    return lam * np.exp(-lam * x) * pdf(lam)


@lru_cache(maxsize=None)
def az(
    x: float,
    pdf: Callable,
    **kwargs,
) -> float:

    return quad_vec(lambda lam: az_integrand(lam, x, pdf), 0, np.inf, **kwargs)[0]


def bz(
    x: float,
    delta: float,
    pdf: Callable,
    **kwargs,
) -> float:

    def integrand(lam: float) -> float:
        return lam * np.exp(-lam * (x + delta)) * pdf(lam)

    return quad_vec(integrand, 0, np.inf, **kwargs)[0]


def cz(
    x: float,
    delta: float,
    pdf: Callable,
    **kwargs,
) -> float:

    def integrand(lam: float) -> float:
        return np.exp(-lam * (x + delta)) * pdf(lam)

    return quad_vec(integrand, 0, np.inf, **kwargs)[0]


def fork_rate(
    proptime: float,
    sum_lambda: float,
    n: int,
    dist: Literal["exp", "log_normal", "lomax"] = "exp",
    hash_std: float = HASH_STD,
    **kwargs,
) -> float:

    if dist == "exp":

        def pdelta_integrand(x: float, delta: float) -> float:
            return (sum_lambda / (n + sum_lambda * x)) ** 2 * (
                n / (n + sum_lambda * (x + delta))
            ) ** n

    else:
        hash_mean = sum_lambda / n
        if dist == "log_normal":
            mu, sigma, _ = calc_ln_params(hash_mean, hash_std)
            # pre-calculate some values for pdf_log_normal
            sigmaSQRRT2PI = sigma * np.sqrt(2 * np.pi)
            twosigsq = 2 * sigma**2

            def pdf(lam: float):
                return pdf_log_normal(
                    lam, mu=mu, sigmaSQRRT2PI=sigmaSQRRT2PI, twosigsq=twosigsq
                )

        elif dist == "lomax":
            lmx_shape, lmx_scale = calc_lmx_params(hash_mean, hash_std)

            def pdf(lam: float):
                return pdf_lomax(lam, lmx_shape=lmx_shape, lmx_scale=lmx_scale)

        def pdelta_integrand(x: float, delta: float) -> float:
            return (
                az(x, pdf, **kwargs)
                * bz(x, delta, pdf, **kwargs)
                * cz(x, delta, pdf, **kwargs) ** (n - 2)
            )

    return (
        n
        * (n - 1)
        * dblquad(
            pdelta_integrand,
            0,
            proptime,
            0,
            np.inf,
        )[0]
    )


if __name__ == "__main__":
    res = fork_rate(
        proptime=EMPIRICAL_PROP_DELAY[0.5],
        sum_lambda=SUM_HASH_RATE,
        n=256,
        hash_std=HASH_STD,
        dist="log_normal",
        epsrel=1e-14,
        limit=155,
    )
    print(res)
