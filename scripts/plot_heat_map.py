import json
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    FIGURES_FOLDER,
    LOG_NORMAL_SIGMA,
    N_MINER,
    SUM_HASH_RATE,
)

with open(DATA_FOLDER / "per_sigmal.pkl", "rb") as f:
    rates_sigma_dict = pickle.load(f)

# transform rates_sigma_dict to a dataframe, parse the keys to named columns and the values to a column
df = pd.DataFrame(
    [
        {
            "distribution": k[0],
            "block_propagation_time": k[1],
            "n": k[2],
            "sumhash": k[3],
            "sigma": k[4],
            "rate": v,
        }
        for k, v in rates_sigma_dict.items()
    ]
)

with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates_integ = json.load(f)

df_integ = pd.DataFrame(rates_integ)
df_integ = df_integ[
    (df_integ["distribution"] == "log_normal") & (df_integ["sumhash"] == SUM_HASH_RATE)
]
df_integ["sigma"] = LOG_NORMAL_SIGMA


df = pd.concat([df, df_integ])
# remove duplicated rows

# for each block propogation time, create a heatmap with x axis being sigma, y axis being n, and color being the fork rate,
# log scale the color and x axis
plt.rcParams.update({"font.size": 18})
for block_propagation_time in [0.763, 8.7, 1000]:
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
        cmap="YlGn",
    )

    # horizontal line at n=19
    plt.axhline(y=N_MINER, color="white", linestyle="--", linewidth=0.5)
    # vertical line at sigma=LOG_NORMAL_SIGMA
    plt.axvline(x=LOG_NORMAL_SIGMA, color="white", linestyle="--", linewidth=0.5)

    # add dot at n=19 and sigma=LOG_NORMAL_SIGMA
    plt.plot(LOG_NORMAL_SIGMA, N_MINER, "*", color="red", markersize=10)

    # plt.yscale("log")
    plt.xscale("log")
    x_ticks = np.logspace(
        np.log10(0.5), np.log10(8), num=5
    )  # Generates 5 ticks from 0.5 to 8, logarithmically spaced
    plt.xticks(x_ticks, [f"{tick:.1f}" for tick in x_ticks])
    cbar = plt.colorbar(label="fork rate $C(\Delta_0)$")
    # add a horizontal line in the colorbar to indicate the fork rate of 0.41
    cbar.ax.hlines(0.0041, xmin=0, xmax=1, colors="blue", linewidth=3)
    # add marker at the fork rate of 0.41
    cbar.ax.plot(0.5, the_fork_rate, "*", color="red", markersize=10)

    plt.xlabel("log normal standard deviation $\sigma$")
    plt.ylabel("number of miners $n$")

    # # highlight the isoline where fork rate is 0.41
    plt.contour(
        df_block_propagation_time.columns,
        df_block_propagation_time.index,
        df_block_propagation_time,
        levels=[0.0041],
        colors="blue",
        linewidths=3,
    )

    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_heatmap_{block_propagation_time}.pdf",
        bbox_inches="tight",
    )
    plt.show()
