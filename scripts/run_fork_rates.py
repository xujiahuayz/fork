import json
import time
from itertools import product

import numpy as np
from scipy.stats import lomax

from fork_env.constants import DATA_FOLDER, SUM_HASH_RATE
from fork_env.simulate import get_fork_rate

exp_dist = lambda n: np.random.exponential(scale=SUM_HASH_RATE / n, size=n)
log_normal_dist = lambda n: np.random.lognormal(
    mean=np.log(SUM_HASH_RATE / n) - np.square(1.11) / 2, sigma=1.11, size=n
)
lomax_dist = lambda n: lomax.rvs(c=1.3, scale=SUM_HASH_RATE * (1.3 - 1) / n, size=n)

distributions = {"exp": exp_dist, "log_normal": log_normal_dist, "lomax": lomax_dist}
block_propagation_times = [0.87, 7.12, 8.7, 10_000]
ns = range(2, 31)
# get distributions and block propagation times combinations
combinations = product(distributions.keys(), block_propagation_times, ns)

start_time = time.time()
rates = []
for distribution, block_propagation_time, n in combinations:
    rate = get_fork_rate(
        repeat=int(1e7),
        n=n,
        hash_distribution=distributions[distribution],
        block_propagation_time=block_propagation_time,
    )
    print(rate)
    rates.append(
        {
            "distribution": distribution,
            "block_propagation_time": block_propagation_time,
            "n": n,
            "rate": rate,
        }
    )

# save rates to json

with open(DATA_FOLDER / "rates_no_sum_constraint.json", "w") as f:
    json.dump(rates, f)

time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")
