import json

import matplotlib.lines as mlines
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

DISTRIBUTIONS = ["exp", "log_normal", "lomax"]
BLOCK_PROPAGATION_TIMES = [0.87, 7.12, 8.7, 1_000]

colors = {
    "exp": "blue",
    "log_normal": "orange",
    "lomax": "green",
}

labels = {
    "exp": "exponential",
    "log_normal": "log normal",
    "lomax": "lomax",
}
# increase font size of plots
plt.rcParams.update({"font.size": 20})
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# export df to excel
for block_propagation_time in BLOCK_PROPAGATION_TIMES:

    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    simulated_handles = []
    analytical_handles = []
    for distribution in DISTRIBUTIONS:

        df_distribution_simulated = df_simulation[
            (df_simulation["distribution"] == distribution)
            & (df_simulation["block_propagation_time"] == block_propagation_time)
        ]

        # Plot simulated data
        plt.plot(
            df_distribution_simulated["n"],
            df_distribution_simulated["rate"],
            label=labels[distribution],
            color=colors[distribution],
            alpha=0.5,  # Making simulated data slightly transparent
            linewidth=5,
        )

        df_distribution = df[
            (df["distribution"] == distribution)
            & (df["block_propagation_time"] == block_propagation_time)
        ]

        # Plot analytical data
        (line,) = plt.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=labels[distribution],
            color=colors[distribution],
            linestyle="--",
            linewidth=2,
        )

        # Store handles for legends
        simulated_handles.append(
            mlines.Line2D([], [], color=colors[distribution], alpha=0.5, linewidth=5)
        )
        analytical_handles.append(line)

    # Create legends without box border outside the plot
    first_legend = plt.legend(
        handles=simulated_handles,
        title="simulated",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.6),
    )
    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=analytical_handles,
        title="analytical",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1, 1.6),
    )

    # set ylimit

    plt.ylim(
        0,
        1.05 * df[df["block_propagation_time"] == block_propagation_time]["rate"].max(),
    )

    # vertical line at n=19
    plt.axvline(x=19, color="black", linestyle=":")

    plt.xlabel("number of miners $n$")
    plt.ylabel("fork rate $C(\Delta_0)$")
    plt.savefig(FIGURES_FOLDER / f"fork_rate_vs_n_{block_propagation_time}.pdf")
    plt.show()
