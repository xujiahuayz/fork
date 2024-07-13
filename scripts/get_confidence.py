import json
import time
from typing import Callable

from joblib import Parallel, delayed
import pandas as pd

# from itertools import product

from fork_env.constants import (
    DATA_FOLDER,
    HASH_STD,
    SIMULATED_FORK_RATES_PATH,
    EMPIRICAL_PROP_DELAY,
    N_MINER,
    SUM_HASHES,
)
from fork_env.integration_lomax import fork_rate_lomax
from fork_env.integration_ln import fork_rate_ln
import numpy as np
from scipy.stats import truncnorm

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate

bis = list(hash_panel["bis"].iloc[-1])

# combinations = product(DIST_KEYS, EMPIRICAL_PROP_DELAY.values(), NUMBER_MINERS_LIST)

start_time = time.time()

def fork_temp(sumhash: float, n:int, hash_mean:float, delta: float, dist: Callable):
    sampled_mean = truncnorm.rvs(
    return dist(
        proptime=delta,
        hash_mean,
        n=N_MINER,
        std=HASH_STD,
    )


# open file for writing by appending
with open(SIMULATED_FORK_RATES_PATH, "a") as f:
    for dist_key, dist in {
        "log_normal": fork_rate_ln,
        "lomax": fork_rate_lomax,
    }.items():
        for sumhash in SUM_HASHES:
            hash_mean = sumhash / N_MINER
            hash_std = sum[]
            for delta in list(EMPIRICAL_PROP_DELAY.values()):



                results = Parallel(n_jobs=-1)(
                    delayed(simulate_fork)(**kwargs) for _ in range(repeat)
                )
        # write to jsonl
        f.write(json.dumps(this_rate) + "\n")


time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")
