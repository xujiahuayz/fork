from scipy.special.cython_special import kn
import numpy as np
from scipy.integrate import quad


def ai_integrand(x: float, lbda: float, bi: float, block_window: int) -> float:
    return (
        lbda
        * np.exp(-lbda * x)
        * block_window
        * np.exp(-((bi - block_window * lbda) ** 2) / (2 * block_window * lbda))
        / (2 * bi * np.exp(bi) * kn(1, bi))
    )


def ai(x: float, bi: float, block_window: int) -> float:
    return quad(
        lambda lbda: ai_integrand(x, lbda, bi, block_window),
        0,
        np.inf,
    )[0]


def ai_alt(x: float, bi: float, block_window: int) -> float:
    return kn(2, bi * np.sqrt(2 * x / block_window + 1)) / (
        kn(1, bi) * (2 * x + block_window)
    )


def p_delta_emp_integrand(
    x: float, delta: float, bis: Iterable[float], block_window: int
) -> float:
    sum_i = 0
    for i, bi in enumerate(bis):
        sum_j = 0
        for j, bj in enumerate(bis):
            if i != j:
                prod_k = prod(
                    kn(1, bk * np.sqrt(2 * (x + delta) / block_window + 1))
                    / (kn(1, bk) * np.sqrt(2 * (x + delta) / block_window + 1))
                    for k, bk in enumerate(bis)
                    if j != k
                )
                sum_j += (
                    kn(2, bj * np.sqrt(2 * (x + delta) / block_window + 1))
                    / (kn(1, bj) * (2 * (x + delta) + block_window))
                    * prod_k
                )
        sum_i += (
            kn(2, bi * np.sqrt(2 * x / block_window + 1))
            / (kn(1, bi) * (2 * x + block_window))
            * sum_j
        )
    return sum_i
