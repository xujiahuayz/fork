from functools import lru_cache

import numpy as np
from scipy.integrate import dblquad, quad_vec

from fork_env.constants import HASH_STD
from fork_env.utils import (
    calc_lmx_shape,
)
from numba import njit


@njit
def az_integrand_lomax(
    lam: float,
    x: float,
    sum_lambda: float,
    n: int,
    c: float,
) -> float:
    return (
        lam
        * n
        / (1 + n * lam / (sum_lambda * (c - 1))) ** c
        / (sum_lambda * (c - 1) + n * lam)
        / np.exp(lam * x)
    )


@lru_cache(maxsize=None)
def az(
    x: float,
    sum_lambda: float,
    n: int,
    c: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_lomax(lam, x, sum_lambda, n, c),
        0,
        np.inf,
        **kwargs,
    )[0]


def bz(
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    c: float,
    **kwargs,
) -> float:

    return az(x + delta, sum_lambda, n, c, **kwargs)


def cz_integrand_lomax(
    lam: float,
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    c: float,
) -> float:
    return az_integrand_lomax(lam, x + delta, sum_lambda, n, c) / lam


def cz(
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    c: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: cz_integrand_lomax(lam, x, delta, sum_lambda, n, c),
        1e-200,
        np.inf,
        **kwargs,
    )[0]


def fork_rate_lomax(
    proptime: float,
    sum_lambda: float,
    n: int,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    hash_mean = sum_lambda / n
    c = calc_lmx_shape(hash_mean, std)
    cn = c**n

    def pdelta_integrand(x: float, delta: float) -> float:
        return (
            az(x, sum_lambda, n, c, **kwargs)
            * bz(x, delta, sum_lambda, n, c, **kwargs)
            * cz(x, delta, sum_lambda, n, c, **kwargs) ** (n - 2)
        ) * cn

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
    res = fork_rate_lomax(
        proptime=16.23,
        sum_lambda=0.002,
        n=42,
        std=0.0002,
    )
    print(res)
