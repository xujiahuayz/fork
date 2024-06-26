import json

import matplotlib.lines as mlines
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from fork_env.constants import (
    FIGURES_FOLDER,
    EMPIRICAL_PROP_DELAY,
    DIST_COLORS,
    DIST_LABELS,
    DIST_KEYS,
    SIMULATED_FORK_RATES_PATH,
    SUM_HASH_RATE,
    N_MINER,
)

from scripts.read_analytical_rates import rates

# load rates from jsonl
with open(SIMULATED_FORK_RATES_PATH, "r") as f:
    rates_simulation = [json.loads(line) for line in f]


rates = rates[(rates["n"] < 500) & (rates["sum_hash"] == SUM_HASH_RATE)]

df = pd.DataFrame(rates)

df_simulation = pd.DataFrame(rates_simulation)


# increase font size of plots
plt.rcParams.update({"font.size": 17})

for block_propagation_time in EMPIRICAL_PROP_DELAY.values():

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    simu_anal = [
        mlines.Line2D(
            [],
            [],
            color="k",
            alpha=1,
            linewidth=5,
            linestyle="--",
            label="analytical $C(\Delta_0)$",
        ),
        mlines.Line2D(
            [],
            [],
            color="k",
            alpha=0.2,
            linewidth=10,
            label="simulated $\\frac{n_{fork}}{n}$",
        ),
    ]
    dist_handle = [
        mlines.Line2D(
            [],
            [],
            color=DIST_COLORS[distribution],
            linewidth=5,
            label=DIST_LABELS[distribution],
        )
        for distribution in DIST_KEYS
    ]
    for distribution in DIST_KEYS:

        df_distribution_simulated = df_simulation[
            (df_simulation["distribution"] == distribution)
            & (df_simulation["block_propagation_time"] == block_propagation_time)
        ]

        # Plot simulated data
        plt.plot(
            df_distribution_simulated["n"],
            df_distribution_simulated["rate"],
            label=DIST_LABELS[distribution],
            color=DIST_COLORS[distribution],
            alpha=0.2,  # Making simulated data slightly transparent
            linewidth=10,
        )

        df_distribution = df[
            (df["distribution"] == distribution)
            & (df["block_propagation_time"] == block_propagation_time)
        ]

        # Plot analytical data
        plt.plot(
            df_distribution["n"],
            df_distribution["rate"],
            color=DIST_COLORS[distribution],
            linestyle="--",
            linewidth=5,
        )

    # Create legends without box border outside the plot with shorter legend handles
    first_legend = plt.legend(
        handles=simu_anal,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        handlelength=0.5,
    )

    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=dist_handle,
        # title="analytical",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1, 0),
        handlelength=0.5,
    )

    # set ylimit
    max_rate = df[df["block_propagation_time"] == block_propagation_time]["rate"].max()
    # plt.ylim(
    #     0,
    #     0.025,
    # )

    # vertical line at n=19
    plt.axvline(x=N_MINER, color="black", linestyle=":")

    plt.xlabel("number of miners $N$")
    plt.ylabel("fork rate")
    # log x axis with base 2
    plt.xscale("log", base=2)

    # make sure the plot is not cut off
    plt.tight_layout()

    plt.savefig(FIGURES_FOLDER / f"fork_rate_vs_n_{block_propagation_time}.pdf")
    plt.show()
