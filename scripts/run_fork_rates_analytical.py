import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import (
    DIST_KEYS,
    EMPIRICAL_PROP_DELAY,
    SUM_HASHES,
    ANALYTICAL_FORK_RATES_PATH_STD,
)
from fork_env.integration import fork_rate


def compute_rate(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n, sumhash, std = args

    the_rate = fork_rate(
        proptime=block_propagation_time,
        sum_lambda=sumhash,
        n=n,
        dist=distribution,
        std=std,
        epsrel=1e-10,
        epsabs=1e-17,
        limit=140,
    )

    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        DIST_KEYS[1:],
        list(EMPIRICAL_PROP_DELAY.values()),
        [8, 10, 15, 20, 50, 100, 150, 200, 500],
        SUM_HASHES,
        [
            # 1e-6,
            # 2e-6,
            # 5e-6,
            # 1e-5,
            2e-5,
            5e-5,
            1e-4,
            2e-4,
            5e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
            2e-2,
        ],
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
