from functools import lru_cache

import numpy as np
from scipy.integrate import quad_vec

from fork_env.constants import HASH_STD
from fork_env.utils import (
    calc_ln_sig,
)
from numba import njit


@njit
def az_integrand_ln(
    lam: float,
    x: float,
    hash_mean: float,
    sigma: float,
) -> float:
    if lam == 0:
        return 0
    sigma2 = sigma**2
    temp = lam / hash_mean
    return np.exp(
        -sigma2 / 8 - pow(np.log(temp), 2) / (2 * sigma2) - lam * x
    ) / np.sqrt(temp)


@lru_cache(maxsize=None)
def az(
    x: float,
    hash_mean: float,
    sigma: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_ln(lam, x, hash_mean, sigma),
        0,
        np.inf,
        **kwargs,
    )[0]


def cz_integrand_ln(
    lam: float,
    x: float,
    delta: float,
    hash_mean: float,
    sigma: float,
) -> float:
    if lam == 0:
        return 0
    return az_integrand_ln(lam, x + delta, hash_mean, sigma) / lam


def cz(
    x: float,
    delta: float,
    hash_mean: float,
    sigma: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: cz_integrand_ln(lam, x, delta, hash_mean, sigma),
        0,
        np.inf,
        **kwargs,
    )[0]


def fork_rate_ln(
    proptime: float,
    n: int,
    sum_lambda: float | None = None,
    hash_mean: float | None = None,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    if hash_mean is None:
        hash_mean = sum_lambda / n

    sigma = calc_ln_sig(hash_mean, std)
    ddlog = n * np.log(np.sqrt(2 * np.pi) * sigma)

    def pdelta_integrand(x: float) -> float:
        return np.exp(
            np.log(az(x, hash_mean, sigma, **kwargs))
            + (n - 1) * np.log(cz(x, proptime, hash_mean, sigma, **kwargs))
            - ddlog
        )

    return 1 - (
        n
        * quad_vec(
            pdelta_integrand,
            0,
            np.inf,
        )[0]
    )


def waste_ln(
    n: int,
    sum_lambda: float | None = None,
    hash_mean: float | None = None,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    if hash_mean is None:
        hash_mean = sum_lambda / n

    sigma = calc_ln_sig(hash_mean, std)
    ddlog = n * np.log(np.sqrt(2 * np.pi) * sigma)

    def pdelta_integrand(x: float) -> float:
        return np.exp(
            2 * np.log(az(x, hash_mean, sigma, **kwargs))
            + (n - 2) * np.log(cz(x, 0, hash_mean, sigma, **kwargs))
            - ddlog
        )

    return (
        n
        * (n - 1)
        * quad_vec(
            pdelta_integrand,
            0,
            np.inf,
        )[0]
    )


if __name__ == "__main__":
    waste = waste_ln(
        hash_mean=0.05 / 38,
        n=38,
        std=0.02,
    )
    print(waste)

    res = fork_rate_ln(
        proptime=14,
        hash_mean=0.05 / 38,
        n=38,
        std=0.02,
    )
    print(res)
