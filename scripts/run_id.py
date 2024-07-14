import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import (
    DATA_FOLDER,
    BLOCK_PROP_TIMES,
    hash_panel_last_row,
    N_MINER,
    SUM_HASHES,
)

from fork_env.integration_empirical import fork_rate_empirical, fork_rate_empirical_id

import pandas as pd

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
BIS = hash_panel_last_row["bis"]


def compute_rate_id(args) -> tuple[tuple, float]:
    iid, block_propagation_time, sum_hash = args

    fork_rate_func = fork_rate_empirical if iid else fork_rate_empirical_id

    the_rate = fork_rate_func(
        proptime=block_propagation_time,
        sum_lambda=sum_hash,
        n=N_MINER,
        bis=BIS,
    )

    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        [True, False],
        BLOCK_PROP_TIMES,
        SUM_HASHES,
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate_id, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(DATA_FOLDER / "analytical_id.pkl", "wb") as f:
        pickle.dump(rates, f)
