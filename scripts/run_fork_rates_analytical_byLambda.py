import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from fork_env.constants import (
    EMPIRICAL_PROP_DELAY,
    SUM_HASH_RATE,
    SUM_HASHES,
    DATA_FOLDER,
    HASH_STD,
    N_MINER,
    # hash_panel_last_row,
)
from fork_env.integration_empirical import fork_rate_empirical
from fork_env.integration_ln import fork_rate_ln
from fork_env.integration_exp import fork_rate_exp
from fork_env.integration_tpl import fork_rate_tpl

dist_dict = {
    "exp": fork_rate_exp,
    # "lomax": fork_rate_lomax,
    "log_normal": fork_rate_ln,
    "trunc_power_law": fork_rate_tpl,
}

# HASH_MEAN = SUM_HASH_RATE / N_MINER

# load hash_panel_last_row
with open(DATA_FOLDER / "hash_panel.pkl", "rb") as f:
    hash_panel = pickle.load(f)

hash_panel_last_row = hash_panel.iloc[-1]


def compute_rate(args) -> tuple[tuple, float]:
    distribution, sum_hash, block_propagation_time, n = args
    try:
        if distribution == "empirical":
            the_rate = fork_rate_empirical(
                proptime=block_propagation_time,
                sum_lambda=sum_hash,
                n=n,
                bis=hash_panel_last_row["bis"],
            )
        if distribution == "exp":
            the_rate = fork_rate_exp(
                proptime=block_propagation_time,
                n=n,
                # hash_mean=HASH_MEAN,
                sum_lambda=sum_hash,
            )
        elif distribution == "log_normal":
            the_rate = fork_rate_ln(
                proptime=block_propagation_time,
                n=n,
                sum_lambda=sum_hash,
                # hash_mean=HASH_MEAN,
                std=HASH_STD,
            )
        elif distribution == "trunc_power_law":
            the_rate = fork_rate_tpl(
                proptime=block_propagation_time,
                n=n,
                sum_lambda=sum_hash,
                # sum_lambda=SUM_HASH_RATE,
                # hash_mean=HASH_MEAN,
                std=HASH_STD,
            )
    except Exception as e:
        # don't care -- too many prints otherwise
        # print(f"Error in {args}: {e}")
        the_rate = np.nan
    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        ["empirical", "exp", "log_normal", "trunc_power_law"],
        SUM_HASHES,
        # log space from 0 .01 to 1000
        np.geomspace(0.01, 1000, num=50),
        [N_MINER],
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(DATA_FOLDER / "rates_analytical_Lambda.pkl", "wb") as f:
        pickle.dump(rates, f)
