# from functools import lru_cache
from functools import lru_cache
import numpy as np
from scipy.integrate import quad_vec
from numba import njit


@njit
def foo(x: float, factor: float) -> float:
    return np.sqrt(1 + 2 * x / factor)


def fork_rate_empirical(
    proptime: float,
    sum_lambda: float,
    n: int,
    bis: list[int],
) -> float:

    factor = np.sum(bis) / sum_lambda

    @lru_cache(maxsize=None)
    def pdelta_integrand(x: float) -> float:
        ax = foo(x, factor)
        cx = foo(x + proptime, factor)
        az = np.mean([np.exp(bi * (1 - ax)) * (1 + bi * ax) for bi in bis]) / (
            factor * ax**3
        )
        cz = np.mean([np.exp(bi * (1 - cx)) for bi in bis]) / cx
        return az * cz ** (n - 1)

    return 1 - (
        n
        * quad_vec(
            pdelta_integrand,
            0,
            np.inf,
        )[0]
    )


def fork_rate_empirical_id(
    proptime: float,
    sum_lambda: float,
    n: int,
    bis: list[int],
) -> float:

    B = np.sum(bis)
    # independent but not identical
    bis = np.array(bis)

    factor = B / sum_lambda

    @lru_cache(maxsize=None)
    def pdelta_integrand(x: float) -> float:
        ax = foo(x, factor)
        cx = foo(x + proptime, factor)
        return np.sum(
            (1 + bis * ax)
            * np.exp(B * (1 - cx) + bis * (cx - ax) - (n - 1) * np.log(cx))
        ) / (factor * ax**3)

    return (
        1
        - quad_vec(
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
    # bis = [x / sum(bis) for x in bis]
    # bis = [4, 1]
    res = fork_rate_empirical(proptime=2, sum_lambda=0.3, n=len(bis), bis=bis)
    print(res)
    res_id = fork_rate_empirical_id(
        proptime=10000, sum_lambda=0.003, n=len(bis), bis=bis
    )
    print(res_id)
