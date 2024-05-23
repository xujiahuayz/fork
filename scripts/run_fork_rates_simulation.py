import json
import time
from itertools import product

import numpy as np

from fork_env.constants import (
    DIST_KEYS,
    HASH_STD,
    SIMULATED_FORK_RATES_PATH,
    SUM_HASH_RATE,
    EMPIRICAL_PROP_DELAY,
    NUMBER_MINERS_LIST,
)
from fork_env.simulate import get_fork_rate
from fork_env.utils import gen_lmx_dist, gen_ln_dist
from scipy.stats import expon


def expon_dist(n_miners: int, sum_hash: float = SUM_HASH_RATE):
    return expon(scale=sum_hash / n_miners)


def lognorm_dist(
    n_miners: int,
    sum_hash: float = SUM_HASH_RATE,
    hash_std: float = HASH_STD,
):
    _, _, ln_dist = gen_ln_dist(hash_mean=sum_hash / n_miners, hash_std=hash_std)
    return ln_dist


def lomax_dist(
    n_miners: int, sum_hash: float = SUM_HASH_RATE, hash_std: float = HASH_STD
) -> np.ndarray:
    _, _, lmx_dist = gen_lmx_dist(hash_mean=sum_hash / n_miners, hash_std=hash_std)
    return lmx_dist


distributions = {
    DIST_KEYS[0]: expon_dist,
    DIST_KEYS[1]: lognorm_dist,
    DIST_KEYS[2]: lomax_dist,
}

combinations = product(DIST_KEYS, EMPIRICAL_PROP_DELAY.values(), NUMBER_MINERS_LIST)

start_time = time.time()
rates = []
# save each line to jsonl
with open(SIMULATED_FORK_RATES_PATH, "w") as f:
    for dist, block_propagation_time, n in combinations:
        # try to catch value error
        try:
            rate = get_fork_rate(
                repeat=int(1e8),
                n_miners=n,
                hash_distribution=lambda n: distributions[dist](n).rvs(size=n),
                block_propagation_time=block_propagation_time,
            )
        except ValueError as e:
            print(f"Error: {e}")
            # next iteration
            continue

        this_rate = {
            "distribution": dist,
            "block_propagation_time": block_propagation_time,
            "n": n,
            "rate": rate,
        }
        print(this_rate)
        # write to jsonl
        f.write(json.dumps(this_rate) + "\n")


time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")
