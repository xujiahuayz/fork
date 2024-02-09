import json

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER, LOG_NORMAL_SIGMA, N_MINER

# Load rates from json
with open(DATA_FOLDER / "rates_integ_log_normal.json", "r") as f:
    rates = json.load(f)

df = pd.DataFrame(rates)

with open(DATA_FOLDER / "rates_integ_log_normal_add.json", "r") as f:
    rates_add = json.load(f)


with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates_integ = json.load(f)

df_integ = pd.DataFrame(rates_integ)
df_integ = df_integ[df_integ["distribution"] == "log_normal"]
# remove the distribution column and add sigma column with value = LOG_NORMAL_SIGMA
df_integ = df_integ.drop(columns=["distribution"])
df_integ["sigma"] = LOG_NORMAL_SIGMA

df_add = pd.DataFrame(rates_add)

df = pd.concat([df, df_add, df_integ])

# for each block propogation time, create a heatmap with x axis being sigma, y axis being n, and color being the fork rate,
# log scale the color and x axis
plt.rcParams.update({"font.size": 18})
for block_propagation_time in [0.86, 8.7, 1000]:
    df_block_propagation_time = df[
        (df["block_propagation_time"] == block_propagation_time)
        & (df["sigma"] > 0.5)
        & (df["sigma"] < 8)
    ]

    df_block_propagation_time = df_block_propagation_time.pivot(
        index="n", columns="sigma", values="rate"
    )
    # get forkrate at sigma=LOG_NORMAL_SIGMA and n=N_MINER
    the_fork_rate = df_block_propagation_time.loc[N_MINER, LOG_NORMAL_SIGMA]

    plt.figure()
    plt.pcolormesh(
        df_block_propagation_time.columns,
        df_block_propagation_time.index,
        df_block_propagation_time,
        shading="auto",
        cmap="viridis",
    )

    # horizontal line at n=19
    plt.axhline(y=N_MINER, color="white", linestyle="--", linewidth=0.5)
    # vertical line at sigma=LOG_NORMAL_SIGMA
    plt.axvline(x=LOG_NORMAL_SIGMA, color="white", linestyle="--", linewidth=0.5)

    # add dot at n=19 and sigma=LOG_NORMAL_SIGMA
    plt.plot(LOG_NORMAL_SIGMA, N_MINER, "+", color="k", markersize=10)

    # plt.yscale("log")
    plt.xscale("log")
    cbar = plt.colorbar(label="fork rate $C(\Delta_0)$")
    # add a horizontal line in the colorbar to indicate the fork rate of 0.41
    cbar.ax.hlines(0.0041, xmin=0, xmax=1, colors="red", linewidth=3)
    # add marker at the fork rate of 0.41
    cbar.ax.plot(0.5, the_fork_rate, "+", color="k", markersize=10)

    plt.xlabel("log normal standard deviation $\sigma$")
    plt.ylabel("number of miners $n$")

    # approximate isolines for the fork rate
    # plt.contour(
    #     df_block_propagation_time.columns,
    #     df_block_propagation_time.index,
    #     df_block_propagation_time,
    #     colors="white",
    #     # linewidths=0.,
    # )
    # # highlight the isoline where fork rate is 0.41
    plt.contour(
        df_block_propagation_time.columns,
        df_block_propagation_time.index,
        df_block_propagation_time,
        levels=[0.0041],
        colors="red",
        linewidths=3,
    )

    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_heatmap_{block_propagation_time}.pdf",
        bbox_inches="tight",
    )
    plt.show()
