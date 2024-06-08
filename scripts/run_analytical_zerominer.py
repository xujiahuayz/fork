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
from fork_env.integration import fork_rate
import pandas as pd
import numpy as np

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate

bis = list(hash_panel["miner_hash"].iloc[-1])


def compute_rate_zerominer(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n_zerominers = args
    bis_with_zero_miners = bis + [0] * n_zerominers
    n = len(bis_with_zero_miners)
    std: float = np.std(bis_with_zero_miners, ddof=0)  # type: ignore

    the_rate = fork_rate(
        proptime=block_propagation_time,
        sum_lambda=SUM_HASH_RATE,
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
        DIST_KEYS,
        BLOCK_PROP_TIMES,
        [0, 10, 20, 50, 100, 150, 200],
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate_zerominer, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(DATA_FOLDER / "analytical_zerominers.pkl", "wb") as f:
        pickle.dump(rates, f)
