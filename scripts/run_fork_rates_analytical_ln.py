import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import numpy as np

from fork_env.constants import (
    EMPIRICAL_PROP_DELAY,
    SUM_HASHES,
    ANALYTICAL_FORK_RATES_PATH_STD,
    HASH_STD,
)
from fork_env.integration_ln import fork_rate_ln

# open ANALYTICAL_FORK_RATES_PATH_STD and load the rates
with open(ANALYTICAL_FORK_RATES_PATH_STD, "rb") as f:
    rates_sigma_dict = pickle.load(f)


def compute_rate(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n, sumhash, std = args
    print(args)
    # if args are in the rates_sigma_dict, return the rate
    if (distribution, block_propagation_time, n, sumhash, std) in rates_sigma_dict:
        the_rate_calculated = rates_sigma_dict[
            (distribution, block_propagation_time, n, sumhash, std)
        ]
        # only use this value if it is not None and not nan and > 1e-20

        the_rate = (
            the_rate_calculated
            if the_rate_calculated and the_rate_calculated > 1e-20
            else fork_rate_ln(
                proptime=block_propagation_time,
                sum_lambda=sumhash,
                n=n,
                std=std,
                # epsrel=1e-10,
                # epsabs=1e-17,
                # limit=140,
            )
        )
    else:
        the_rate = fork_rate_ln(
            proptime=block_propagation_time,
            sum_lambda=sumhash,
            n=n,
            std=std,
            # epsrel=1e-10,
            # epsabs=1e-17,
            # limit=140,
        )

    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        ["log_normal"],
        list(EMPIRICAL_PROP_DELAY.values()),
        [10, 12, 15, 20, 30, 50, 70, 100, 120, 150, 200, 250, 300],
        SUM_HASHES,
        [
            3e-5,
            4e-5,
            5e-5,
            8e-5,
            # 1e-4,
            1.5e-4,
            2e-4,
            5e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
            1.5e-2,
        ]
        + [HASH_STD],
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(ANALYTICAL_FORK_RATES_PATH_STD, "wb") as f:
        pickle.dump(rates, f)
