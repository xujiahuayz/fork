import numpy as np
from scipy.integrate import quad_vec
from fork_env.utils import calc_ex_rate

from numba import njit


def fork_rate_exp(
    proptime: float,
    n: int,
    sum_lambda: float | None = None,
    hash_mean: float | None = None,
) -> float:
    if hash_mean is None:
        hash_mean = sum_lambda / n

    r = calc_ex_rate(hash_mean=hash_mean)

    @njit
    def pdelta_integrand(x: float) -> float:
        return np.exp(
            n * np.log(r) - 2 * np.log(r + x) - (n - 1) * np.log(proptime + r + x)
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
    res = fork_rate_exp(
        proptime=14.916,
        sum_lambda=0.005,
        n=38,
    )
    print(res)
