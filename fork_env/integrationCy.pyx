from functools import lru_cache
from libc.math cimport exp, log, sqrt, pow
import numpy as np
from scipy.integrate import dblquad, quad_vec
cimport numpy as np

# Define constants
cdef double SQRRT2PI = sqrt(2 * np.pi)

cdef double pdf_log_normal(double lam, double sum_lambda, double n, double sigma):
    """
    pdf of the log normal distribution where mean = np.log(sum_lambda / n) - np.square(sigma) / 2 and sigma = sigma
    """
    return 1 / (lam * sigma * SQRRT2PI * exp(
        ((pow(sigma,2) / 2 + log(lam*n/sum_lambda)) ** 2) / (2 * pow(sigma,2))
    ))



cdef double pdf_lomax(double lam, double sum_lambda, double n, double c):
    """
    pdf of the Lomax distribution where mean = np.log(sum_lambda / n) - np.square(sigma) / 2 and sigma = sigma
    """
    return (
        n
        * c
        / pow(1 + n * lam / (sum_lambda * (c - 1)), c)
        / (sum_lambda * (c - 1) + n * lam)
    )

