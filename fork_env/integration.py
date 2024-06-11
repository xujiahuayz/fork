from functools import lru_cache
from typing import Callable, Literal

import numpy as np
from scipy.integrate import dblquad, quad_vec

from fork_env.constants import EMPIRICAL_PROP_DELAY, HASH_STD, SUM_HASH_RATE
from fork_env.utils import (
    calc_ln_sig,
    calc_lmx_shape,
)


SQRRT2PI = np.sqrt(2 * np.pi)


@lru_cache(maxsize=None)
def pdf_log_normal(lam: float, sum_lambda: float, n: int, sigma: float) -> float:
    """
    pdf of the log normal distribution where mean = np.log(sum_lambda / n) - np.square(sigma) / 2 and sigma = sigma
    """
    sigma2 = sigma**2
    # capture warnings - if there is a warning, pause

    return pow(
        lam
        * SQRRT2PI
        * np.exp(
            pow(sigma2 / 2 + np.log(lam * n / sum_lambda), 2) / (2 * sigma2)
            + np.log(sigma)
        ),
        -1,
    )


@lru_cache(maxsize=None)
def pdf_lomax(lam: float, sum_lambda: float, n: int, c: float) -> float:
    """
    pdf of the lomax distribution where scale = sum_lambda * (c - 1) / n and c = c
    """
    return (
        n
        * c
        * (1 + n * lam / (sum_lambda * (c - 1))) ** (-c)
        / (sum_lambda * (c - 1) + n * lam)
    )


def az_integrand(
    lam: float,
    x: float,
    sum_lambda: float,
    n: int,
    pdf: Callable,
) -> float:
    return lam * np.exp(-lam * x) * pdf(lam, sum_lambda, n)


@lru_cache(maxsize=None)
def az(
    x: float,
    sum_lambda: float,
    n: int,
    pdf: Callable,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: az_integrand(lam, x, sum_lambda, n, pdf), 0, np.inf, **kwargs
    )[0]


def bz(
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    pdf: Callable,
    **kwargs,
) -> float:

    def integrand(lam: float) -> float:
        return lam * np.exp(-lam * (x + delta)) * pdf(lam, sum_lambda, n)

    return quad_vec(integrand, 0, np.inf, **kwargs)[0]


def cz(
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    pdf: Callable,
    **kwargs,
) -> float:

    def integrand(lam: float) -> float:
        return np.exp(-lam * (x + delta)) * pdf(lam, sum_lambda, n)

    return quad_vec(integrand, 0, np.inf, **kwargs)[0]


def fork_rate(
    proptime: float,
    sum_lambda: float,
    n: int,
    dist: Literal["exp", "log_normal", "lomax"] = "exp",
    std: float = HASH_STD,
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
            sigma = calc_ln_sig(hash_mean, std)

            def pdf(lam: float, sum_lambda: float, n: int):
                return pdf_log_normal(lam, sum_lambda, n, sigma=sigma)

        elif dist == "lomax":
            c = calc_lmx_shape(hash_mean, std)

            def pdf(lam: float, sum_lambda: float, n: int):
                return pdf_lomax(lam, sum_lambda, n, c=c)

        def pdelta_integrand(x: float, delta: float) -> float:
            return (
                az(x, sum_lambda, n, pdf, **kwargs)
                * bz(x, delta, sum_lambda, n, pdf, **kwargs)
                * cz(x, delta, sum_lambda, n, pdf, **kwargs) ** (n - 2)
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
        proptime=0.816,
        sum_lambda=0.0005,
        n=10,
        std=8e-05,
        dist="log_normal",
        # epsrel=1e-15,
        # epsabs=1e-17,
        # limit=250,
    )
    print(res)
