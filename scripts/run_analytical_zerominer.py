import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import (
    DATA_FOLDER,
    DIST_KEYS,
    SUM_HASH_RATE,
    BLOCK_PROP_TIMES,
    hash_panel_last_row,
)
from fork_env.integration_exp import fork_rate_exp
from fork_env.integration_lomax import fork_rate_lomax
from fork_env.integration_ln import fork_rate_ln
from fork_env.integration_empirical import fork_rate_empirical, fork_rate_empirical_id

import pandas as pd
import numpy as np

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")


def compute_rate_zerominer(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n_zerominers = args
    # if args in rates:
    #     return args, rates[args]
    bis_with_zero_miners: list[int] = (
        list(hash_panel_last_row["bis"]) + [0] * n_zerominers
    )
    n = len(bis_with_zero_miners)
    if distribution.startswith("empirical"):
        for_rate_func = (
            fork_rate_empirical_id
            if distribution == "empirical_id"
            else fork_rate_empirical
        )
        the_rate = for_rate_func(
            proptime=block_propagation_time,
            sum_lambda=SUM_HASH_RATE,
            n=n,
            bis=bis_with_zero_miners,
        )
    else:
        hash_with_zero_miners = (
            list(hash_panel_last_row["miner_hash"]) + [0] * n_zerominers
        )
        mean: float = np.mean(hash_with_zero_miners)
        std: float = np.std(hash_with_zero_miners, ddof=0)  # type: ignore
        if distribution == "exp":
            the_rate = fork_rate_exp(
                proptime=block_propagation_time,
                n=n,
                hash_mean=mean,
            )
        elif distribution == "log_normal":
            the_rate = fork_rate_ln(
                proptime=block_propagation_time,
                n=n,
                hash_mean=mean,
                std=std,
            )

    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        ["empirical_id", "empirical_iid"] + DIST_KEYS,
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

    with open(DATA_FOLDER / "analytical_zerominers.pkl", "wb") as f:
        pickle.dump(rates, f)
