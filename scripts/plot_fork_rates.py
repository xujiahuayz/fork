from itertools import product
import time
import numpy as np
from scipy.stats import lomax

from fork_env.simulate import get_fork_rate

exp_dist = lambda n: np.random.exponential(scale=11_400, size=n)
log_normal_dist = lambda n: np.random.lognormal(mean=-9.96, sigma=1.11, size=n)
lomax_dist = lambda n: lomax.rvs(c=1.3, scale=2.6e-5, size=n)

distributions = [exp_dist, log_normal_dist, lomax_dist]
block_propagation_times = [0.87, 7.12, 8.7, 10_000]

# get distributions and block propagation times combinations
combinations = product(distributions, block_propagation_times)


start_time = time.time()
rates = []
for distribution, block_propagation_time in combinations:
    rate = get_fork_rate(
        repeat=99_999,
        n=19,
        hash_distribution=distribution,
        block_propagation_time=block_propagation_time,
    )
    print(rate)
    rates.append(
        {
            "distribution": distribution,
            "block_propagation_time": block_propagation_time,
            "rate": rate,
        }
    )
time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")
