import json

import matplotlib.lines as mlines
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from fork_env.constants import (
    DATA_FOLDER,
    FIGURES_FOLDER,
    EMPIRICAL_PROP_DELAY,
    DIST_COLORS,
    DIST_LABELS,
    DIST_KEYS,
)

# load rates from json
with open(DATA_FOLDER / "rates_no_sum_constraint.json", "r") as f:
    rates_simulation = json.load(f)

with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates = json.load(f)


df = pd.DataFrame(rates)

df_simulation = pd.DataFrame(rates_simulation)


# increase font size of plots
plt.rcParams.update({"font.size": 17})

# export df to excel
for block_propagation_time in EMPIRICAL_PROP_DELAY.values():

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    simu_anal = [
        mlines.Line2D(
            [], [], color="k", alpha=1, linewidth=5, linestyle="--", label="analytical"
        ),
        mlines.Line2D([], [], color="k", alpha=0.2, linewidth=10, label="simulated"),
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
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.05),
    )
    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=dist_handle,
        # title="analytical",
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.02, -0.05),
    )

    # set ylimit
    max_rate = df[df["block_propagation_time"] == block_propagation_time]["rate"].max()
    plt.ylim(
        -0.05 * max_rate,
        1.05 * max_rate,
    )

    # vertical line at n=19
    plt.axvline(x=19, color="black", linestyle=":")

    plt.xlabel("number of miners $N$")
    plt.ylabel("fork rate $C(\Delta_0)$")

    # make sure the plot is not cut off
    plt.tight_layout()

    plt.savefig(FIGURES_FOLDER / f"fork_rate_vs_n_{block_propagation_time}.pdf")
    plt.show()
