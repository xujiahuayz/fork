import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    EMPIRICAL_PROP_DELAY,
    FIGURES_FOLDER,
    SUM_HASHES,
    N_MINER,
    HASH_STD,
)

with open(DATA_FOLDER / "rates_analytical_std_lomax.pkl", "rb") as f:
    rates_sigma_dict = pickle.load(f)


# transform rates_sigma_dict to a dataframe, parse the keys to named columns and the values to a column
df = pd.DataFrame(
    [
        {
            "distribution": k[0],
            "block_propagation_time": k[1],
            "n": k[2],
            "sumhash": k[3],
            "std": k[4],
            "rate": v,
        }
        for k, v in rates_sigma_dict
    ]
)

df = df[df["distribution"] == "lomax"]

# for each block propogation time, create a heatmap with x axis being sigma, y axis being n, and color being the fork rate,
# log scale the color and x axis
plt.rcParams.update({"font.size": 18})
# resize the plot
plt.rcParams["figure.figsize"] = [5, 5]

x_ticks = np.logspace(
    np.log10(0.5), np.log10(8), num=5
)  # Generates 5 ticks from 0.5 to 8, logarithmically spaced

for block_propagation_time in list(EMPIRICAL_PROP_DELAY.values()):
    for sum_hash in SUM_HASHES:
        df_block_propagation_time = df[
            ((df["block_propagation_time"] == block_propagation_time))
            & (df["sumhash"] == sum_hash)
        ]
        this_df_block_propagation_time = df_block_propagation_time[
            (df_block_propagation_time["std"] > 7e-5)
            & (df_block_propagation_time["std"] < 1.5e-3)
            & (df_block_propagation_time["n"] < 400)
            & (df_block_propagation_time["n"] > 6)
        ]

        this_df_block_propagation_time = this_df_block_propagation_time.pivot(
            index="n", columns="std", values="rate"
        )
        # this_df_block_propagation_time = df_block_propagation_time.apply(
        #     pd.to_numeric, errors="coerce"
        # )

        # get forkrate at sigma=LOG_NORMAL_SIGMA and n=N_MINER
        the_fork_rate = this_df_block_propagation_time.loc[N_MINER, HASH_STD]

        plt.figure()
        plt.pcolormesh(
            this_df_block_propagation_time.columns,
            this_df_block_propagation_time.index,
            this_df_block_propagation_time,
            shading="auto",
            cmap="YlGn",
        )

        # horizontal line at n=19
        plt.axhline(y=N_MINER, color="white", linestyle="--", linewidth=0.5)
        # vertical line at sigma=LOG_NORMAL_SIGMA
        plt.axvline(x=HASH_STD, color="white", linestyle="--", linewidth=0.5)

        # add dot at n=19 and sigma=LOG_NORMAL_SIGMA
        plt.plot(HASH_STD, N_MINER, "*", color="red", markersize=10)

        plt.xscale("log")
        # plt.yscale("log")
        # log y with base 2
        plt.yscale("log", base=2)
        # plt.xticks(x_ticks, [f"{tick:.1f}" for tick in x_ticks])
        cbar = plt.colorbar(label="fork rate $C(\Delta_0)$", location="top")
        # make cbar ticker labels scientific
        cbar.formatter.set_powerlimits((0, 0))

        # add a horizontal line in the colorbar to indicate the fork rate of 0.41
        cbar.ax.vlines(0.0041, ymin=0, ymax=1, colors="blue", linewidth=3)
        # add marker at the fork rate of 0.41
        cbar.ax.plot(the_fork_rate, 0.5, "*", color="red", markersize=10)

        plt.xlabel("standard deviation $s$")
        plt.ylabel("number of miners $N$")

        # # highlight the isoline where fork rate is 0.41
        plt.contour(
            this_df_block_propagation_time.columns,
            this_df_block_propagation_time.index,
            this_df_block_propagation_time,
            levels=[0.0041],
            colors="blue",
            linewidths=3,
        )

        plt.savefig(
            FIGURES_FOLDER
            / f"fork_rate_heatmap_{block_propagation_time}_{sum_hash}_lomax.pdf",
            bbox_inches="tight",
        )
        plt.show()
