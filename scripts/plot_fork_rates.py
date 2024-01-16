import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import lomax

from fork_env.simulate import get_fork_rate

from matplotlib import pyplot as plt

exp_dist = lambda n: np.random.exponential(scale=11_400, size=n)
log_normal_dist = lambda n: np.random.lognormal(mean=-9.96, sigma=1.11, size=n)
lomax_dist = lambda n: lomax.rvs(c=1.3, scale=2.6e-5, size=n)

distributions = {"exp": exp_dist, "log_normal": log_normal_dist, "lomax": lomax_dist}
block_propagation_times = [0.87, 7.12, 8.7, 10_000]
ns = range(2, 31)
# get distributions and block propagation times combinations
combinations = product(distributions.keys(), block_propagation_times, ns)

start_time = time.time()
rates = []
for distribution, block_propagation_time, n in combinations:
    rate = get_fork_rate(
        repeat=999_999,
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
time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")

# for each block propagation time, plot the fork rate as y-axis and n as x-axis for each distribution as line type
df = pd.DataFrame(rates)
for block_propagation_time in block_propagation_times:
    fig, ax = plt.subplots(figsize=(10, 10))
    df_block_propagation_time = df[
        df["block_propagation_time"] == block_propagation_time
    ]
    for distribution in distributions.keys():
        df_distribution = df_block_propagation_time[
            df_block_propagation_time["distribution"] == distribution
        ]
        ax.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=f"{distribution}, {block_propagation_time}",
        )

    ax.legend()
    ax.set_xlabel("n")
    ax.set_ylabel("fork rate")
    # show plot
    plt.show()
