import json

import matplotlib.lines as mlines
import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER

# load rates from json
with open(DATA_FOLDER / "rates_no_sum_constraint.json", "r") as f:
    rates_simulation = json.load(f)

with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates = json.load(f)


df = pd.DataFrame(rates)

df_simulation = pd.DataFrame(rates_simulation)

DISTRIBUTIONS = ["exp", "log_normal", "lomax"]
BLOCK_PROPAGATION_TIMES = [0.87, 7.12, 8.7, 1_000]

colors = ["blue", "orange", "green", "red"]

labels = {
    "exp": "exponential",
    "log_normal": "log normal",
    "lomax": "lomax",
}
# increase font size of plots
plt.rcParams.update({"font.size": 20})

# export df to excel
for distribution in DISTRIBUTIONS:

    simulated_handles = []
    analytical_handles = []
    for block_propagation_time in BLOCK_PROPAGATION_TIMES:

        df_distribution_simulated = df_simulation[
            (df_simulation["distribution"] == distribution)
            & (df_simulation["block_propagation_time"] == block_propagation_time)
        ]

        this_color = colors[BLOCK_PROPAGATION_TIMES.index(block_propagation_time)]

        # Plot simulated data
        plt.plot(
            df_distribution_simulated["n"],
            df_distribution_simulated["rate"],
            label=f"$\Delta_0 = {block_propagation_time}$",
            color=this_color,
            alpha=0.2,  # Making simulated data slightly transparent
            linewidth=10,
        )

        df_distribution = df[
            (df["distribution"] == distribution)
            & (df["block_propagation_time"] == block_propagation_time)
        ]

        # Plot analytical data
        (line,) = plt.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=f"$\Delta_0 = {block_propagation_time}$",
            color=this_color,
            linestyle="--",
            linewidth=5,
        )

        # Store handles for legends
        simulated_handles.append(
            mlines.Line2D([], [], color=this_color, alpha=0.2, linewidth=5)
        )
        analytical_handles.append(line)

    # Create legends without box border outside the plot
    first_legend = plt.legend(
        handles=simulated_handles,
        title="simulated",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.35, 1),
    )
    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=analytical_handles,
        title="analytical",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.8, 1),
    )

    # set ylimit
    # log y-axis
    plt.yscale("log")
    plt.ylim(1e-4, 1)

    # vertical line at n=19
    plt.axvline(x=19, color="black", linestyle=":")

    plt.xlabel("number of miners $n$")
    plt.ylabel("fork rate $C(\Delta_0)$")
    # make sure the plot is not cut off
    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_vs_n_{distribution}.pdf", bbox_inches="tight"
    )
    plt.show()
