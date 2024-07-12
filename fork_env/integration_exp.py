import numpy as np
from scipy.integrate import quad_vec
from fork_env.utils import calc_ex_rate

from numba import njit


def fork_rate_exp(
    proptime: float,
    sum_lambda: float,
    n: int,
) -> float:

    r = calc_ex_rate(hash_mean=sum_lambda / n)

    @njit
    def pdelta_integrand(x: float) -> float:
        return 1 / (r + x) ** 2 / (proptime + r + x) ** (n - 1)

    return 1 - (
        n
        * r**n
        * quad_vec(
            pdelta_integrand,
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
