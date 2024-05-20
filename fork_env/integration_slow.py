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
def pdf_lognormal_component(lam: float, mu: float, twosigsq: float) -> float:
    return (np.log(lam) - mu) ** 2 / twosigsq


def az_integrand_lognormal(
    lam: float,
    x: float,
    mu: float,
    twosigsq: float,
) -> float:
    return np.exp(-lam * x - pdf_lognormal_component(lam, mu, twosigsq))


@lru_cache(maxsize=None)
def az_lognormal(
    x: float,
    mu: float,
    twosigsq: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_lognormal(lam, x, mu, twosigsq), 0, np.inf, **kwargs
    )[0]


def bz_integrand_lognormal(
    lam: float,
    x: float,
    delta: float,
    mu: float,
    twosigsq: float,
) -> float:
    return az_integrand_lognormal(lam, x + delta, mu, twosigsq)


# @lru_cache(maxsize=None)
def bz_lognormal(
    x: float,
    delta: float,
    mu: float,
    twosigsq: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: bz_integrand_lognormal(lam, x, delta, mu, twosigsq),
        0,
        np.inf,
        **kwargs,
    )[0]


def cz_integrand_lognormal(
    lam: float,
    x: float,
    delta: float,
    mu: float,
    twosigsq: float,
) -> float:
    return bz_integrand_lognormal(lam, x, delta, mu, twosigsq) / lam


# @lru_cache(maxsize=None)
def cz_lognormal(
    x: float,
    delta: float,
    mu: float,
    twosigsq: float,
    **kwargs,
) -> float:

    return quad_vec(
        lambda lam: cz_integrand_lognormal(lam, x, delta, mu, twosigsq),
        0,
        np.inf,
        **kwargs,
    )[0]


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

            def pdelta_integrand(x: float, delta: float) -> float:
                return (
                    az_lognormal(x, mu=mu, twosigsq=twosigsq, **kwargs)
                    * bz_lognormal(x, delta, mu=mu, twosigsq=twosigsq, **kwargs)
                    * cz_lognormal(x, delta, mu=mu, twosigsq=twosigsq, **kwargs)
                    ** (n - 2)
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
            ) / sigmaSQRRT2PI**n

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


"""
{'distribution': 'log_normal', 'block_propagation_time': 0.816, 'n': 256, 'rate': 0.0011218}

0.0011281368182624815
old:
0.001112057385745302

old:
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
0.0011301264148556565


# {'distribution': 'log_normal', 'block_propagation_time': 0.816, 'n': 128, 'rate': 0.0011678}
#     res = fork_rate(
#         proptime=EMPIRICAL_PROP_DELAY[0.5],
#         sum_lambda=SUM_HASH_RATE,
#         n=128,
#         hash_std=HASH_STD,
#         dist="log_normal",
#         epsrel=1e-9,
#         epsabs=1e-16,
#         limit=130,
#     )
#     print(res)
#     28.9s
# 0.0011592981474107603

# res = fork_rate(
#     proptime=EMPIRICAL_PROP_DELAY[0.5],
#     sum_lambda=SUM_HASH_RATE,
#     n=128,
#     hash_std=HASH_STD,
#     dist="log_normal",
#     epsrel=1e-10,
#     # epsabs=1e-19,
#     limit=130,
# )
# print(res)
# 0.0011593012106172903

# {'distribution': 'log_normal', 'block_propagation_time': 0.816, 'n': 2, 'rate': 0.000697}

# res = fork_rate(
#     proptime=EMPIRICAL_PROP_DELAY[0.5],
#     sum_lambda=SUM_HASH_RATE,
#     n=2,
#     hash_std=HASH_STD,
#     dist="log_normal",
#     epsrel=1e-9,
#     epsabs=1e-16,
#     limit=130,
# )
# print(res)
# 0.0006924438523584548

# res = fork_rate(
#     proptime=EMPIRICAL_PROP_DELAY[0.5],
#     sum_lambda=SUM_HASH_RATE,
#     n=2,
#     hash_std=HASH_STD,
#     dist="log_normal",
#     epsrel=1e-13,
#     epsabs=1e-19,
#     limit=130,
# )
# print(res)
# 0.0006930817261667343

# res = fork_rate(
#     proptime=EMPIRICAL_PROP_DELAY[0.5],
#     sum_lambda=SUM_HASH_RATE,
#     n=2,
#     hash_std=HASH_STD,
#     dist="log_normal",
#     epsrel=1e-16,
#     epsabs=1e-22,
#     limit=380,
# )
# print(res)
# 0.0006930931657026595

# """
