import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import (
    DIST_KEYS,
    BLOCK_PROP_TIMES,
    NUMBER_MINERS_LIST,
    SUM_HASHES,
    ANALYTICAL_FORK_RATES_PATH,
    HASH_STD,
)
from fork_env.integration import fork_rate


def compute_rate(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n, sumhash = args

    the_rate = fork_rate(
        proptime=block_propagation_time,
        sum_lambda=sumhash,
        n=n,
        dist=distribution,
        std=HASH_STD,
        epsrel=1e-9,
        epsabs=1e-16,
        limit=130,
    )

    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(DIST_KEYS, BLOCK_PROP_TIMES, NUMBER_MINERS_LIST, SUM_HASHES)

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(ANALYTICAL_FORK_RATES_PATH, "wb") as f:
        pickle.dump(rates, f)
