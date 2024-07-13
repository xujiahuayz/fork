import fork_env.integration_ln as int_ln
import fork_env.integration_lomax as int_lomax
import fork_env.integration_empirical as int_empirical
from fork_env.utils import calc_ln_sig, calc_lmx_shape
import numpy as np
from scipy.integrate import dblquad
from functools import lru_cache
from numba import njit


@njit
def az_exp(rate: float, x: float) -> float:
    return rate / pow(rate + x, 2)


@njit
def bz_exp(rate: float, x: float, delta: float) -> float:
    return az_exp(rate, x + delta)


@njit
def cz_exp(rate: float, x: float, delta: float) -> float:
    return rate / (rate + x + delta)


def fork_rate_mixed(
    proptime: float,
    sum_lambda: float,
    n_zero: int,
    bis: list[int],
    main_dist: str,
) -> float:
    n_nonzero = len(bis)
    n = n_zero + n_nonzero
    B = np.sum(bis)
    factor = B / sum_lambda
    rate = factor / 2
    if main_dist == "empirical":

        ints = [zele(bi, B) for bi in bis]

        @lru_cache(maxsize=None)
        def pdf_p(lbda: float):
            lbda = lbda * factor
            return pdf_empirical(lbda, bis, ints)

        az_main = lambda x: int_empirical.az(x, sum_lambda, pdf_p) * factor
        bz_main = (
            lambda x, delta: int_empirical.bz(x, delta, sum_lambda, pdf_p) * factor
        )
        cz_main = (
            lambda x, delta: int_empirical.cz(x, delta, sum_lambda, pdf_p) * factor
        )

    else:
        hash_mean = float(np.mean(bis)) / factor

        if main_dist == "exp":
            main_rate = 1 / hash_mean

            az_main = lambda x: az_exp(main_rate, x)
            bz_main = lambda x, delta: bz_exp(main_rate, x, delta)
            cz_main = lambda x, delta: cz_exp(main_rate, x, delta)

        else:
            std = float(np.std(bis, ddof=0)) / factor

            if main_dist == "log_normal":
                sigma = calc_ln_sig(hash_mean=hash_mean, hash_std=std)
                dd = np.sqrt(2 * np.pi) * sigma
                az_main = lambda x: int_ln.az(x, sum_lambda, n, sigma) / dd
                bz_main = (
                    lambda x, delta: int_ln.bz(x, delta, sum_lambda, n, sigma) / dd
                )
                cz_main = (
                    lambda x, delta: int_ln.cz(x, delta, sum_lambda, n, sigma) / dd
                )
            elif main_dist == "lomax":
                c = calc_lmx_shape(hash_mean=hash_mean, hash_std=std)
                az_main = lambda x: int_lomax.az(x, sum_lambda, n, c) * c
                bz_main = lambda x, delta: int_lomax.bz(x, delta, sum_lambda, n, c) * c
                cz_main = lambda x, delta: int_lomax.cz(x, delta, sum_lambda, n, c) * c

    def pdelta_integrand(x: float, delta: float) -> float:
        return (
            ((n_nonzero * az_main(x) + n_zero * az_exp(rate, x)) / n)
            * ((n_nonzero * bz_main(x, delta) + n_zero * bz_exp(rate, x, delta)) / n)
            * ((n_nonzero * cz_main(x, delta) + n_zero * cz_exp(rate, x, delta)) / n)
            ** (n - 2)
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
    res = fork_rate_mixed(
        proptime=14.916,
        sum_lambda=0.00171,
        n_zero=0,
        bis=bis,
        main_dist="empirical",
    )
    print(res)
