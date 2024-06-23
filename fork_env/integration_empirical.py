from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.integrate import dblquad, quad_vec

from fork_env.utils import zele, pdf_empirical


def az_integrand_empirical(lam: float, x: float, pdf: Callable) -> float:
    return lam * np.exp(-lam * x) * pdf(lam)


@lru_cache(maxsize=None)
def az(
    x: float,
    sum_lambda: float,
    pdf: Callable,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_empirical(lam, x, pdf),
        0,
        sum_lambda,
    )[0]


def bz(
    x: float,
    delta: float,
    sum_lambda: float,
    pdf: Callable,
) -> float:

    return az(x + delta, sum_lambda, pdf)


def cz_integrand_lomax(
    lam: float,
    x: float,
    delta: float,
    pdf: Callable,
) -> float:
    return az_integrand_empirical(lam, x + delta, pdf) / lam


def cz(
    x: float,
    delta: float,
    sum_lambda: float,
    pdf: Callable,
) -> float:

    return quad_vec(lambda lam: cz_integrand_lomax(lam, x, delta, pdf), 0, sum_lambda)[
        0
    ]


def fork_rate_empirical(
    proptime: float,
    sum_lambda: float,
    n: int,
    bis: list[int],
) -> float:
    B = np.sum(bis)
    factor = B / sum_lambda

    ints = [zele(bi, B) for bi in bis]

    @lru_cache(maxsize=None)
    def pdf_p(lbda: float):
        lbda = lbda * factor
        return pdf_empirical(lbda, bis, ints)

    def pdelta_integrand(x: float, delta: float) -> float:
        return (
            (az(x, sum_lambda, pdf_p) * factor)
            * (bz(x, delta, sum_lambda, pdf_p) * factor)
            * (cz(x, delta, sum_lambda, pdf_p) * factor) ** (n - 2)
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
    bis = [
        8704,
        7633,
        3485,
        3230,
        2007,
        1324,
        1141,
        311,
        306,
        295,
        265,
        259,
        256,
        241,
        169,
        137,
        56,
        28,
        26,
        24,
        23,
        18,
        12,
        11,
        7,
        6,
        4,
        4,
        3,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
    ]
    res = fork_rate_empirical(proptime=14.916, sum_lambda=0.00171, n=len(bis), bis=bis)
    print(res)
