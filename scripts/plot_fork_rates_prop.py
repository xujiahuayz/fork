import json

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER

# load rates from json
with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates = json.load(f)


df = pd.DataFrame(rates)

DISTRIBUTIONS = ["exp", "log_normal", "lomax"]
BLOCK_PROPAGATION_TIMES = [0.87, 7.12, 8.7, 1_000]
NUM_MINERS = sorted([2, 5, 19, 30], reverse=True)

colors = ["blue", "red", "orange", "green"]

labels = {
    "exp": "exponential",
    "log_normal": "log normal",
    "lomax": "lomax",
}
# increase font size of plots
plt.rcParams.update({"font.size": 20})

# export df to excel
for distribution in DISTRIBUTIONS:

    for n_miner in NUM_MINERS:

        df_distribution = df[
            (df["distribution"] == distribution) & (df["n"] == n_miner)
        ]

        # Plot analytical data
        plt.plot(
            df_distribution["block_propagation_time"],
            df_distribution["rate"],
            label=n_miner,
            color=colors[NUM_MINERS.index(n_miner)],
            alpha=0.8,
            linewidth=2,
        )

    plt.legend(
        title="$n$",
        frameon=False,
        loc="upper left",
        # bbox_to_anchor=(1, 1),
    )

    # set ylimit
    # log y-axis
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(0, 1)

    plt.xlabel("block propagation time $\Delta_0$")
    plt.ylabel("fork rate $C(\Delta_0)$")
    # make sure the plot is not cut off
    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_time_{distribution}.pdf", bbox_inches="tight"
    )
    plt.show()
