import json
import time
from itertools import product

import pandas as pd

from fork_env.constants import (
    SIMULATED_FORK_RATES_EMP_DIST,
    EMPIRICAL_PROP_DELAY,
    DATA_FOLDER,
    BLOCK_WINDOW,
)
from fork_env.simulate import get_fork_rate
from fork_env.utils import EmpDist

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate

bis = hash_panel["miner_hash"].iloc[-1]


zero_miners = [0, 10, 50, 100]

combinations = product(EMPIRICAL_PROP_DELAY.values(), zero_miners)

start_time = time.time()
# check if file exists
rates = []
try:
    with open(SIMULATED_FORK_RATES_EMP_DIST, "r") as f:
        for line in f:
            # load json
            data = json.loads(line)
            # append to list
            rates.append(data)
except FileNotFoundError:
    pass

# open file for writing by appending
with open(SIMULATED_FORK_RATES_EMP_DIST, "a") as f:
    for block_propagation_time, n_zero_miner in combinations:
        # check if it has already been computed
        if any(
            rate["block_propagation_time"] == block_propagation_time
            and rate["n_zero_miner"] == n_zero_miner
            for rate in rates
        ):
            print(f"Already computed: {block_propagation_time}, {n_zero_miner}")
            continue
        # try to catch value error
        try:
            miner_bi = bis.to_list() + [0] * n_zero_miner
            emp_dist = EmpDist(bis=miner_bi)
            rate = get_fork_rate(
                repeat=int(5e3),
                n_miners=len(miner_bi),
                hash_distribution=lambda n: emp_dist.rvs(size=n) / BLOCK_WINDOW,
                block_propagation_time=block_propagation_time,
            )
        except ValueError as e:
            print(f"Error: {e}")
            # next iteration
            continue

        this_rate = {
            "block_propagation_time": block_propagation_time,
            "n_zero_miner": n_zero_miner,
            "rate": rate,
        }
        print(this_rate)
        # write to jsonl
        f.write(json.dumps(this_rate) + "\n")


time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")
