import json

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER

# load rates from json
with open(DATA_FOLDER / "rates_no_sum_constraint.json", "r") as f:
    rates_simulation = json.load(f)

with open(DATA_FOLDER / "rates_integration.json", "r") as f:
    rates = json.load(f)

with open(DATA_FOLDER / "rates_integration_exp.json", "r") as f:
    rates_exp = json.load(f)
# for each block propagation time, plot the fork rate as y-axis and n as x-axis for each distribution as line type
df = pd.DataFrame(rates)
# remove rows with exp
df = df[df["distribution"] != "exp"]

df_exp = pd.DataFrame(rates_exp)
df = pd.concat([df_exp, df])

df_simulation = pd.DataFrame(rates_simulation)

colors = {
    "exp": "blue",
    "log_normal": "orange",
    "lomax": "green",
}
# export df to excel
for block_propagation_time in df["block_propagation_time"].unique():
    for distribution in df["distribution"].unique():
        df_distribution = df[
            (df["distribution"] == distribution)
            & (df["block_propagation_time"] == block_propagation_time)
        ]
        df_distribution_simulated = df_simulation[
            (df_simulation["distribution"] == distribution)
            & (df_simulation["block_propagation_time"] == block_propagation_time)
        ]

        # plot simulated data with transparent color and thicker line
        # and then plot the analytical data with a thinner line but same color
        plt.plot(
            df_distribution_simulated["n"],
            df_distribution_simulated["rate"],
            label=f"{distribution} simulated",
            color=colors[distribution],
            alpha=0.2,
            linewidth=5,
        )
        plt.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=f"{distribution} analytical",
            color=colors[distribution],
            linestyle="--",
            linewidth=1,
        )
        # set y axis limits
        # plt.ylim(0, 0.018)
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("fork rate")
    plt.title("fork rate vs n")
    plt.savefig(FIGURES_FOLDER / f"fork_rate_vs_n_{block_propagation_time}.pdf")
    plt.show()
