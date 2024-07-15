import json
import time
from typing import Callable

from joblib import Parallel, delayed
import pandas as pd

# from itertools import product

from fork_env.constants import (
    DATA_FOLDER,
    EMPIRICAL_PROP_DELAY,
    N_MINER,
    # SUM_HASHES,
    SUM_HASH_RATE,
    hash_panel_last_row,
    BLOCK_WINDOW,
)
from fork_env.integration_lomax import fork_rate_lomax
from fork_env.integration_ln import fork_rate_ln
import numpy as np

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate

bis = hash_panel_last_row["bis"]
ps = [b / BLOCK_WINDOW for b in bis]
var_pi = [p * (1 - p) / BLOCK_WINDOW for p in ps]
sum_var_p = sum(var_pi)
sum_var_p2 = sum([v**2 for v in var_pi])


start_time = time.time()


def fork_temp(
    mean_mean: float,
    mean_std: float,
    mean_var: float,
    var_std: float,
    delta: float,
    dist: Callable,
):
    sampled_mean = np.random.normal(mean_mean, mean_std)
    sampled_var = np.random.normal(mean_var, var_std)
    return dist(
        proptime=delta,
        hash_mean=sampled_mean,
        n=N_MINER,
        std=np.sqrt(sampled_var),
    )


if __name__ == "__main__":
    # open file for writing by appending
    with open(DATA_FOLDER / "rates_confidence.jsonl", "a") as f:
        for dist_key, dist in {
            "log_normal": fork_rate_ln,
            "lomax": fork_rate_lomax,
        }.items():
            for sumhash in [SUM_HASH_RATE]:
                hash_mean = sumhash / N_MINER
                hash_var = (
                    sum([(p * sumhash) ** 2 for p in ps]) - sumhash**2 / N_MINER
                ) / (N_MINER - 1)
                mean_std = np.sqrt(sum_var_p) * hash_mean
                var_std = (
                    np.sqrt(2 / (N_MINER * (N_MINER - 1)) * sum_var_p2) * hash_mean
                )

                for delta in list(EMPIRICAL_PROP_DELAY.values()) + [8.7]:
                    results = Parallel(n_jobs=-1)(
                        delayed(fork_temp)(
                            hash_mean, mean_std, hash_var, var_std, delta, dist
                        )
                        for _ in range(int(1e2))
                    )
                    this_rate = {
                        "dist": dist_key,
                        "sumhash": sumhash,
                        "proptime": delta,
                        "mean_mean": hash_mean,
                        "mean_std": mean_std,
                        "mean_var": hash_var,
                        "var_std": var_std,
                        "results": results,
                    }
                    print(this_rate)

                    f.write(json.dumps(this_rate) + "\n")

    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
