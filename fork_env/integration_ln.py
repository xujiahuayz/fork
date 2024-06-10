from functools import lru_cache

import numpy as np
from scipy.integrate import dblquad, quad_vec

from fork_env.constants import EMPIRICAL_PROP_DELAY, HASH_STD, SUM_HASH_RATE
from fork_env.utils import (
    calc_ln_sig,
    calc_lmx_shape,
)


@lru_cache(maxsize=None)
def integrand_com(sigma2: float, lam: float, sum_lambda: float, n: int) -> float:
    return pow(sigma2 / 2 + np.log(lam * n / sum_lambda), 2) / (2 * sigma2)


@lru_cache(maxsize=None)
def az_integrand_ln(
    lam: float,
    x: float,
    sum_lambda: float,
    n: int,
    sigma: float,
) -> float:
    sigma2 = sigma**2
    return pow(
        np.exp(
            integrand_com(
                sigma2=sigma2,
                lam=lam,
                sum_lambda=sum_lambda,
                n=n,
            )
            + lam * x
        ),
        -1,
    )


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


def bz(
    x: float,
    delta: float,
    sum_lambda: float,
    n: int,
    sigma: float,
    **kwargs,
) -> float:

    return az(x + delta, sum_lambda, n, sigma, **kwargs)


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


# def pdelta_integrand_ln(
#     x: float,
#     delta: float,
#     sum_lambda: float,
#     n: int,
#     sigma: float,
#     **kwargs,
# ) -> float:

#     return (
#         az(x, sum_lambda, n, sigma, **kwargs)
#         * bz(x, delta, sum_lambda, n, sigma, **kwargs)
#         * cz(x, delta, sum_lambda, n, sigma, **kwargs) ** (n - 2)
#     )


def fork_rate_ln(
    proptime: float,
    sum_lambda: float,
    n: int,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    hash_mean = sum_lambda / n
    sigma = calc_ln_sig(hash_mean, std)
    dd = np.sqrt(2 * np.pi) * sigma
    # if pow(SQRRT2PI * sigma, n) has overflow, use the following
    try:
        ddpow = pow(dd, n)

        def pdelta_integrand(x: float, delta: float) -> float:
            return (
                az(x, sum_lambda, n, sigma, **kwargs)
                * bz(x, delta, sum_lambda, n, sigma, **kwargs)
                * cz(x, delta, sum_lambda, n, sigma, **kwargs) ** (n - 2)
            )

    except OverflowError:
        ddpow = 1

        def pdelta_integrand(x: float, delta: float) -> float:
            return (
                (az(x, sum_lambda, n, sigma, **kwargs) / dd)
                * (bz(x, delta, sum_lambda, n, sigma, **kwargs) / dd)
                * (cz(x, delta, sum_lambda, n, sigma, **kwargs) / dd) ** (n - 2)
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
    ) / ddpow


if __name__ == "__main__":
    res = fork_rate_ln(
        proptime=0.816,
        sum_lambda=0.0005,
        n=20,
        std=0.02,
        # epsrel=1e-4,
        # epsabs=1e-6,
        # limit=50,
    )
    print(res)
