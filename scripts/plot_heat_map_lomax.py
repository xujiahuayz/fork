import pickle

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    FIGURES_FOLDER,
    LOMAX_C,
    N_MINER,
    SUM_HASH_RATE,
)

with open(DATA_FOLDER / "per_c_new.pkl", "rb") as f:
    rates_c_dict = pickle.load(f)

# transform rates_c_dict to a dataframe, parse the keys to named columns and the values to a column
df = pd.DataFrame(
    [
        {
            "distribution": k[0],
            "block_propagation_time": k[1],
            "n": k[2],
            "sumhash": k[3],
            "c": k[4],
            "rate": v,
        }
        for k, v in rates_c_dict.items()
    ]
)

with open(DATA_FOLDER / "rates_analytical.pkl", "rb") as f:
    rates = pickle.load(f)

df_integ = pd.DataFrame(
    [
        {
            "distribution": k[0][0],
            "block_propagation_time": k[0][1],
            "n": k[0][2],
            "sumhash": k[0][3],
            "rate": k[1],
        }
        for k in rates
    ]
)
df_integ = df_integ[
    (df_integ["distribution"] == "lomax") & (df_integ["sumhash"] == SUM_HASH_RATE)
]
df_integ["c"] = LOMAX_C


df = pd.concat([df, df_integ])
# remove duplicated rows

# make c 1/c
df["c"] = 1 / df["c"]

# for each block propogation time, create a heatmap with x axis being c, y axis being n, and color being the fork rate,
# log scale the color and x axis
plt.rcParams.update({"font.size": 18})
# resize the plot
plt.rcParams["figure.figsize"] = [5, 5]

# x_ticks = [1, 2, 4, 8, 16, 32]

for block_propagation_time in [0.763, 2.48, 8.7, 16.472, 1000]:
    df_block_propagation_time = df[
        (df["block_propagation_time"] == block_propagation_time)
    ]

    df_block_propagation_time = df_block_propagation_time.pivot(
        index="n", columns="c", values="rate"
    )
    df_block_propagation_time.interpolate(
        method="linear", axis=0, inplace=True
    )  # Row-wise interpolation
    df_block_propagation_time.interpolate(
        method="linear", axis=1, inplace=True
    )  # Column-wise interpolation
    # get forkrate at c=LOG_NORMAL_c and n=N_MINER
    the_fork_rate = df_block_propagation_time.loc[N_MINER, 1 / LOMAX_C]

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
    # vertical line at c=LOG_NORMAL_c
    plt.axvline(x=1 / LOMAX_C, color="white", linestyle="--", linewidth=0.5)

    # add dot at n=19 and c=LOG_NORMAL_c
    plt.plot(1 / LOMAX_C, N_MINER, "*", color="red", markersize=10)

    plt.xlim([1 / 1e25, 1.005])

    # plt.xscale("log")

    # place xticks at 1, 2, 4, 8 with exactly 1 decimal place
    # plt.xticks(x_ticks, x_ticks)

    cbar = plt.colorbar(label="fork rate $C(\Delta_0)$", location="top")
    # make cbar ticker labels scientific
    cbar.formatter.set_powerlimits((0, 0))

    # add a horizontal line in the colorbar to indicate the fork rate of 0.41
    cbar.ax.vlines(0.0041, ymin=0, ymax=1, colors="blue", linewidth=3)
    # add marker at the fork rate of 0.41
    cbar.ax.plot(the_fork_rate, 0.5, "*", color="red", markersize=10)

    plt.xlabel("lomax $\\alpha^{-1}$")
    plt.ylabel("number of miners $N$")

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
        FIGURES_FOLDER / f"fork_rate_heatmap_c_{block_propagation_time}.pdf",
        bbox_inches="tight",
    )
    plt.show()
