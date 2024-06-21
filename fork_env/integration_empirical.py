from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.integrate import dblquad, quad_vec

from numba import njit


@njit
def ccdf_component(lbda: float, bi: float):
    if bi == 0:
        return 1 / np.exp(lbda / 2)
    return np.exp(bi - pow(bi, 2) / (2 * lbda) - lbda / 2)


def az_integrand_lomax(lam: float, x: float, pdf: Callable) -> float:

    return lam * np.exp(-lam * x) * pdf(lam)


@lru_cache(maxsize=None)
def az(
    x: float,
    sum_lambda: float,
    pdf: Callable,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_lomax(lam, x, pdf),
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
    return az_integrand_lomax(lam, x + delta, pdf) / lam


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
    bis: list[int],
) -> float:
    n = len(bis)
    B = np.sum(bis)
    factor = B / sum_lambda

    ints = [
        (
            quad_vec(lambda x: ccdf_component(x, bi), 0, bi)[0]
            + quad_vec(lambda x: ccdf_component(x, bi), bi, B)[0]
        )
        for bi in bis
    ]

    @lru_cache(maxsize=None)
    def pdf_p(lbda: float):
        lbda = lbda * factor
        # break down integration into two parts to avoid singularities
        return np.mean([ccdf_component(lbda, bis[i]) / w for i, w in enumerate(ints)])

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

    res = fork_rate_empirical(
        proptime=50,
        sum_lambda=1 / 600,
        bis=[
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
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    print(res)
