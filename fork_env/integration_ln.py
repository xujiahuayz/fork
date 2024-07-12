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
    sum_lambda: float,
    n: int,
    sigma: float,
) -> float:
    sigma2 = sigma**2
    temp = lam * n / sum_lambda
    return np.exp(
        -sigma2 / 8 - pow(np.log(temp), 2) / (2 * sigma2) - lam * x
    ) / np.sqrt(temp)


@lru_cache(maxsize=None)
def az(
    x: float,
    sum_lambda: float,
    n: int,
    sigma: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_ln(lam, x, sum_lambda, n, sigma),
        0,
        np.inf,
        **kwargs,
    )[0]


def cz_integrand_ln(
    lam: float,
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    sigma: float,
) -> float:
    return az_integrand_ln(lam, x + delta, sum_lambda, n, sigma) / lam


def cz(
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    sigma: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: cz_integrand_ln(lam, x, delta, sum_lambda, n, sigma),
        0,
        np.inf,
        **kwargs,
    )[0]


def fork_rate_ln(
    proptime: float,
    sum_lambda: float,
    n: int,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    hash_mean = sum_lambda / n
    sigma = calc_ln_sig(hash_mean, std)
    ddlog = n * np.log(np.sqrt(2 * np.pi) * sigma)

    def pdelta_integrand(x: float) -> float:
        return np.exp(
            np.log(az(x, sum_lambda, n, sigma, **kwargs))
            + (n - 1) * np.log(cz(x, proptime, sum_lambda, n, sigma, **kwargs))
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


if __name__ == "__main__":
    res = fork_rate_ln(
        proptime=0,
        sum_lambda=0.0171,
        n=38,
        std=0.02,
    )
    print(res)
