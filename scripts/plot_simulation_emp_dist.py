# open SIMULATED_FORK_RATES_EMP_DIST

# Path: scripts/plot_simulation_emp_dist.py

import json

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    SIMULATED_FORK_RATES_EMP_DIST,
    FIGURES_FOLDER,
    EMPRITICAL_FORK_RATE,
)

with open(SIMULATED_FORK_RATES_EMP_DIST, "r") as f:
    rates = [json.loads(line) for line in f]

df = pd.DataFrame(rates)

plt.rcParams.update({"font.size": 20})

fig, ax = plt.subplots(figsize=(8, 6))

for block_propagation_time in df["block_propagation_time"].unique():

    df_distribution = df[(df["block_propagation_time"] == block_propagation_time)]

    ax.plot(
        df_distribution["n_zero_miner"],
        df_distribution["rate"],
        marker="o",
        label=f"{block_propagation_time}",
    )

# add horizontal line at 00.41
ax.axhline(EMPRITICAL_FORK_RATE, color="black", linestyle="--")

ax.set_xlabel("number of 0-block miners")
ax.set_ylabel("fork rate $C(\Delta_0)$")
ax.legend(title="$\Delta_0$")

# save to file
plt.savefig(FIGURES_FOLDER / "fork_rate_emp_dist.pdf", bbox_inches="tight")
