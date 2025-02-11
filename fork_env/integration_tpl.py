import numpy as np
from scipy.integrate import quad_vec

from fork_env.constants import HASH_STD
from fork_env.utils import (
    calc_truncpl_params,
)


def fork_rate_tpl(
    proptime: float,
    n: int,
    sum_lambda: float | None = None,
    hash_mean: float | None = None,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    if hash_mean is None:
        hash_mean = sum_lambda / n
    alpha, beta = calc_truncpl_params(hash_mean=hash_mean, hash_std=std)

    def pdelta_integrand(x: float) -> float:
        return (beta + x) ** (-2 + alpha) * (beta + proptime + x) ** (
            (alpha - 1) * (n - 1)
        )

    return (
        1
        - n
        * (1 - alpha)
        * beta ** (n * (1 - alpha))
        * quad_vec(
            pdelta_integrand,
            0,
            np.inf,
        )[0]
    )


def waste_tpl(
    n: int,
    sum_lambda: float | None = None,
    hash_mean: float | None = None,
    std: float = HASH_STD,
    **kwargs,
) -> float:
    if hash_mean is None:
        hash_mean = sum_lambda / n
    alpha, beta = calc_truncpl_params(hash_mean=hash_mean, hash_std=std)

    return n * (n - 1) * (1 - alpha) ** 2 / (beta + beta * n - alpha * beta * n)


if __name__ == "__main__":
    waste_tpl(
        sum_lambda=0.0017,
        n=33,
        std=0.000135,
    )

    proptime = 5
    res = (
        fork_rate_tpl(
            proptime=proptime,
            sum_lambda=0.0017,
            n=33,
            std=0.000135,
        )
        # / proptime
    )
    print(res)
