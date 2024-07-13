from functools import lru_cache

import numpy as np
from scipy.integrate import quad_vec

from fork_env.constants import HASH_STD
from fork_env.utils import (
    calc_lmx_shape,
)
from numba import njit


@njit
def az_integrand_lomax(
    lam: float,
    x: float,
    hash_mean: float,
    c: float,
) -> float:
    return (
        lam
        / (1 + lam / (hash_mean * (c - 1))) ** c
        / (hash_mean * (c - 1) + lam)
        / np.exp(lam * x)
    )


@lru_cache(maxsize=None)
def az(
    x: float,
    hash_mean: float,
    c: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_lomax(lam, x, hash_mean, c),
        0,
        np.inf,
        **kwargs,
    )[0]


@njit
def cz_integrand_lomax(
    lam: float,
    x: float,
    delta: float,
    hash_mean: float,
    c: float,
) -> float:
    return (
        1
        / (1 + lam / (hash_mean * (c - 1))) ** c
        / (hash_mean * (c - 1) + lam)
        / np.exp(lam * (x + delta))
    )


def cz(
    x: float,
    delta: float,
    hash_mean: float,
    c: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: cz_integrand_lomax(lam, x, delta, hash_mean, c),
        0,
        np.inf,
        **kwargs,
    )[0]


def fork_rate_lomax(
    proptime: float,
    n: int,
    sum_lambda: float | None = None,
    hash_mean: float | None = None,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    if hash_mean is None:
        hash_mean = sum_lambda / n
    c = calc_lmx_shape(hash_mean, std)
    nlogc = n * np.log(c)

    def pdelta_integrand(x: float) -> float:
        return np.exp(
            np.log(az(x, hash_mean, c, **kwargs))
            + (n - 1) * np.log(cz(x, proptime, hash_mean, c, **kwargs))
            + nlogc
        )

    return (
        1
        - n
        * quad_vec(
            pdelta_integrand,
            0,
            np.inf,
        )[0]
    )


if __name__ == "__main__":
    res = fork_rate_lomax(
        proptime=1,
        hash_mean=0.31 / 42,
        n=42,
        std=6,
    )
    print(res)
