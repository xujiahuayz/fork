from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.stats import expon, lognorm, lomax

from fork_env.constants import LOG_NORMAL_SIGMA, LOMAX_C, SUM_HASH_RATE

# from numba import jit, njit


@lru_cache(maxsize=None)
def pdf_exp(lam: float, sum_lambda: float, n: int) -> float:
    """
    pdf of the exponential distribution where scale = sum_lambda / n
    """
    return expon.pdf(lam, scale=sum_lambda / n)  # type: ignore


@lru_cache(maxsize=None)
def pdf_log_normal(lam: float, sum_lambda: float, n: int, sigma: float) -> float:
    """
    pdf of the log normal distribution where mean = np.log(sum_lambda / n) - np.square(sigma) / 2 and sigma = sigma
    """
    return lognorm.pdf(
        lam, s=sigma, scale=np.exp(np.log(sum_lambda / n) - np.square(sigma) / 2)
    )  # type: ignore


@lru_cache(maxsize=None)
def pdf_lomax(lam: float, sum_lambda: float, n: int, c: float) -> float:
    """
    pdf of the lomax distribution where scale = sum_lambda * (c - 1) / n and c = c
    """
    return lomax.pdf(lam, c=c, scale=sum_lambda * (c - 1) / n)  # type: ignore


def fork_rate(
    proptime: float,
    sum_lambda: float,
    n: int,
    dist: Literal["exp", "log_normal", "lomax"] = "exp",
    **kwargs,
) -> float:

    if dist == "exp":

        def pdelta(delta: float) -> float:
            # mu = n / sum_lambda

            def integrand(x: float) -> float:
                return (sum_lambda / (n + sum_lambda * x)) ** 2 * (
                    n / (n + sum_lambda * (x + delta))
                ) ** n

                # return 1 / ((mu + x) ** 2 * (mu + x + delta) ** n)

            return quad(integrand, 0, np.inf, **kwargs)[0]

    else:

        if dist == "log_normal":
            if "sigma" in kwargs:
                sigma = kwargs["sigma"]
                # remove sigma from kwargs
                kwargs.pop("sigma")
            else:
                sigma = LOG_NORMAL_SIGMA
            pdf = lambda lam, sum_lambda, n: pdf_log_normal(
                lam, sum_lambda, n, sigma=sigma
            )
        elif dist == "lomax":
            if "c" in kwargs:
                c = kwargs["c"]
                # remove c from kwargs
                kwargs.pop("c")
            else:
                c = LOMAX_C
            pdf = lambda lam, sum_lambda, n: pdf_lomax(lam, sum_lambda, n, c=c)

        def az(
            x: float,
            sum_lambda: float,
            n: int,
            **kwargs,
        ) -> float:

            def integrand(lam: float) -> float:
                return lam * np.exp(-lam * x) * pdf(lam, sum_lambda, n)

            return quad(integrand, 0, np.inf, **kwargs)[0]

        def bz(
            x: float,
            delta: float,
            sum_lambda: float,
            n: int,
            **kwargs,
        ) -> float:

            def integrand(lam: float) -> float:
                return lam * np.exp(-lam * (x + delta)) * pdf(lam, sum_lambda, n)

            return quad(integrand, 0, np.inf, **kwargs)[0]

        def cz(
            x: float,
            delta: float,
            sum_lambda: float,
            n: int,
            **kwargs,
        ) -> float:

            def integrand(lam: float) -> float:
                return np.exp(-lam * (x + delta)) * pdf(lam, sum_lambda, n)

            return quad(integrand, 0, np.inf, **kwargs)[0]

        def pdelta(delta: float) -> float:

            def integrand(x: float) -> float:
                return (
                    az(x, sum_lambda, n, **kwargs)
                    * bz(x, delta, sum_lambda, n, **kwargs)
                    * cz(x, delta, sum_lambda, n, **kwargs) ** (n - 2)
                )

            return quad(integrand, 0, np.inf, **kwargs)[0]

    return n * (n - 1) * quad(pdelta, 0, proptime, epsrel=5e-4)[0]


if __name__ == "__main__":

    result = fork_rate(
        proptime=0.87,
        sum_lambda=SUM_HASH_RATE,
        n=4,
        dist="log_normal",
        epsrel=1e-9,
        epsabs=1e-16,
        limit=130,
        limlst=10,
    )

    kwargs = {"epsrel": 1e-9, "epsabs": 1e-16, "limit": 130, "limlst": 10}
