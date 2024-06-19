import numpy as np
from scipy.integrate import dblquad

from numba import njit


def fork_rate_exp(
    proptime: float,
    sum_lambda: float,
    n: int,
) -> float:

    @njit
    def pdelta_integrand(x: float, delta: float) -> float:
        return (sum_lambda / (n + sum_lambda * x)) ** 2 * (
            n / (n + sum_lambda * (x + delta))
        ) ** n

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
    res = fork_rate_exp(
        proptime=14.916,
        sum_lambda=0.005,
        n=38,
    )
    print(res)
