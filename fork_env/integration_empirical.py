from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.integrate import quad_vec


def az_integrand_empirical(lam: float, x: float, pdf: Callable) -> float:
    return lam * np.exp(-lam * x) * pdf(lam)


@lru_cache(maxsize=None)
def az(
    x: float,
    pdf: Callable,
) -> float:

    return quad_vec(
        lambda lam: az_integrand_empirical(lam, x, pdf),
        0,
        np.inf,
    )[0]


def cz_integrand_lomax(
    lam: float,
    x: float,
    delta: float,
    pdf: Callable,
) -> float:
    return np.exp(-lam * (x + delta)) * pdf(lam)


def cz(
    x: float,
    delta: float,
    pdf: Callable,
) -> float:

    return quad_vec(lambda lam: cz_integrand_lomax(lam, x, delta, pdf), 0, np.inf)[0]


def fork_rate_empirical(
    proptime: float,
    sum_lambda: float,
    n: int,
    bis: list[int],
) -> float:
    B = np.sum(bis)

    @lru_cache(maxsize=None)
    def pdf_p(lbda: float):
        def plb(l: float, b: float) -> float:
            ll = l * B / sum_lambda
            return np.exp(-pow(b - ll, 2) / (2 * ll)) / np.sqrt(
                2 * np.pi * l * sum_lambda / B
            )

        return np.mean([plb(lbda, bi) for bi in bis])

    def pdelta_integrand(x: float) -> float:
        return (az(x, pdf_p)) * (cz(x, proptime, pdf_p)) ** (n - 1)

    return 1 - (
        n
        * quad_vec(
            pdelta_integrand,
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
    bis = [4, 1]
    res = fork_rate_empirical(proptime=2, sum_lambda=0.3, n=len(bis), bis=bis)
    print(res)
