import json
import time
from itertools import product

from fork_env.constants import (
    DIST_KEYS,
    SIMULATED_FORK_RATES_PATH,
    EMPIRICAL_PROP_DELAY,
    NUMBER_MINERS_LIST,
)
from fork_env.simulate import get_fork_rate
from fork_env.utils import (
    expon_dist,
    lognorm_dist,
    lomax_dist,
)

distributions = {
    DIST_KEYS[0]: expon_dist,
    DIST_KEYS[1]: lognorm_dist,
    DIST_KEYS[2]: lomax_dist,
}

combinations = product(DIST_KEYS, EMPIRICAL_PROP_DELAY.values(), NUMBER_MINERS_LIST)

start_time = time.time()
# check if file exists
rates = []
try:
    with open(SIMULATED_FORK_RATES_PATH, "r") as f:
        for line in f:
            # load json
            data = json.loads(line)
            # append to list
            rates.append(data)
except FileNotFoundError:
    pass

# open file for writing by appending
with open(SIMULATED_FORK_RATES_PATH, "a") as f:
    for dist, block_propagation_time, n in combinations:
        # check if it has already been computed
        if any(
            rate["distribution"] == dist
            and rate["block_propagation_time"] == block_propagation_time
            and rate["n"] == n
            for rate in rates
        ):
            print(f"Already computed: {dist}, {block_propagation_time}, {n}")
            continue
        # try to catch value error
        try:
            rate = get_fork_rate(
                repeat=int(5e7),
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
