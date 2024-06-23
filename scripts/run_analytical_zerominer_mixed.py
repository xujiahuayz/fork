import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import (
    DATA_FOLDER,
    DIST_KEYS,
    SUM_HASH_RATE,
    BLOCK_PROP_TIMES,
)
from fork_env.integration_mixed import fork_rate_mixed

import pandas as pd

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate

BIS = list(hash_panel["bis"].iloc[-1])
# check if the file exists
try:
    with open(DATA_FOLDER / "analytical_zerominers_mixed.pkl", "rb") as f:
        rates = pickle.load(f)
        rates = dict(rates)
except FileNotFoundError:
    rates = {}


def compute_rate_zerominer(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n_zerominers = args
    if args in rates:
        return args, rates[args]
    the_rate = fork_rate_mixed(
        proptime=block_propagation_time,
        sum_lambda=SUM_HASH_RATE,
        n_zero=n_zerominers,
        bis=BIS,
        main_dist=distribution,
    )

    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        ["empirical"] + DIST_KEYS,
        BLOCK_PROP_TIMES,
        [0, 10, 20, 50, 100, 150, 200, 300],
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate_zerominer, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(DATA_FOLDER / "analytical_zerominers_mixed.pkl", "wb") as f:
        pickle.dump(rates, f)
