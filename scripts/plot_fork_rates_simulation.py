import json
import pickle

import matplotlib.lines as mlines
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from fork_env.constants import (
    FIGURES_FOLDER,
    EMPIRICAL_PROP_DELAY,
    DIST_COLORS,
    DIST_LABELS,
    SIMULATED_FORK_RATES_PATH,
    N_MINER,
    DATA_FOLDER,
    EMPRITICAL_FORK_RATE,
)
import numpy as np
import gzip

dist_keys = list(DIST_COLORS.keys())
# load rates from jsonl
with gzip.open(SIMULATED_FORK_RATES_PATH, "rt") as f:
    rates_simulation = [json.loads(line) for line in f]

rates = pd.DataFrame(
    list(w[0]) + [w[1]]
    for w in pickle.load(open(DATA_FOLDER / "rates_analytical_line.pkl", "rb"))
)
rates.columns = ["distribution", "block_propagation_time", "n", "rate"]
# sort by n
rates = rates.sort_values(by="n")

df = pd.DataFrame(rates)

df_simulation = pd.DataFrame(rates_simulation)


# increase font size of plots
plt.rcParams.update({"font.size": 17})

simu_anal = [
    mlines.Line2D(
        [],
        [],
        color="k",
        alpha=1,
        linewidth=3,
        # linestyle="--",
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
        linestyle="--" if distribution == "empirical" else "-",
        linewidth=1 if distribution == "empirical" else 3,
        label=DIST_LABELS[distribution],
    )
    for distribution in dist_keys
]

for block_propagation_time in list(EMPIRICAL_PROP_DELAY.values()) + [8.7]:
    df_simulation["rate"] = df_simulation["time_diffs"].apply(
        lambda x: np.mean(np.array(x) < block_propagation_time)
    )

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    for distribution in dist_keys + ["empirical"]:

        df_distribution_simulated = df_simulation[
            (df_simulation["distribution"] == distribution)
        ]

        if distribution != "empirical":
            # Plot simulated data
            plt.plot(
                df_distribution_simulated["n"],
                df_distribution_simulated["rate"],
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
            linestyle="--" if distribution == "empirical" else "-",
            linewidth=1 if distribution == "empirical" else 3,
        )

    # Create legends without box border outside the plot with shorter legend handles
    first_legend = plt.legend(
        handles=simu_anal,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        handlelength=0.8,
    )

    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=dist_handle,
        # title="analytical",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1, 0),
        handlelength=0.8,
    )

    # set ylimit
    max_rate = df[df["block_propagation_time"] == block_propagation_time]["rate"].max()

    plt.axvline(x=N_MINER, color="black", linestyle=":")
    # write N_MINER next to the vertical line
    plt.text(N_MINER + 1, 0.00012, f"$N={N_MINER}$")
    # horizontal line at y=EMPRITICAL_FORK_RATE
    plt.axhline(y=EMPRITICAL_FORK_RATE, color="black", linestyle=":")
    # write empirical fork rate next to the horizontal line
    plt.text(2**8 * 0.5, 0.002, "0.41%")

    plt.xlabel("number of miners $N$")
    plt.ylabel("fork rate")

    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlim(7, 500)
    plt.ylim(0.0001, 1)
    # title
    plt.title(f"$\Delta_0 = {block_propagation_time}$ [s]")

    # make sure the plot is not cut off
    plt.tight_layout()

    plt.savefig(FIGURES_FOLDER / f"fork_rate_vs_n_{block_propagation_time}.pdf")
    plt.show()
