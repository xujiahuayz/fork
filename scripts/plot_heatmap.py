import json
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    EMPIRICAL_PROP_DELAY,
    FIGURES_FOLDER,
    SUM_HASH_RATE,
    N_MINER,
)
from scripts.get_confidence import ps
from fork_env.integration_lomax import fork_rate_lomax
from fork_env.integration_ln import fork_rate_ln

with open(DATA_FOLDER / "rates_confidence.jsonl", "r") as f:
    rates_confidence = [json.loads(line) for line in f]

with open(DATA_FOLDER / "rates_analytical_heatmap.pkl", "rb") as f:
    rates_all = pickle.load(f)

# log scale the color and x axis
plt.rcParams.update({"font.size": 18})
# resize the plot
plt.rcParams["figure.figsize"] = [5, 5]

x_ticks = np.logspace(
    np.log10(0.5), np.log10(8), num=5
)  # Generates 5 ticks from 0.5 to 8, logarithmically spaced

for key, dict_items in {
    "lomax": fork_rate_lomax,
    "log_normal": fork_rate_ln,
}.items():
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
            for k, v in rates_all
            if k[0] == key
        ]
    )

    for block_propagation_time in list(EMPIRICAL_PROP_DELAY.values()) + [8.7]:
        df_block_propagation_time = df[
            ((df["block_propagation_time"] == block_propagation_time))
            & (df["sumhash"] == SUM_HASH_RATE)
        ]
        this_df_block_propagation_time = df_block_propagation_time

        this_df_block_propagation_time = this_df_block_propagation_time.pivot(
            index="n", columns="std", values="rate"
        )

        confidence_rates = [
            rate["results"]
            for rate in rates_confidence
            if rate["dist"] == key
            and rate["proptime"] == block_propagation_time
            and rate["sumhash"] == SUM_HASH_RATE
        ][0]
        # get 5th percentile and 95th percentile of confidence_rates = [0.0334, 0.434..]
        p5, p95 = np.percentile(confidence_rates, [5, 95])
        this_mean = SUM_HASH_RATE / N_MINER
        this_std = np.sqrt(
            (sum([(p * SUM_HASH_RATE) ** 2 for p in ps]) - N_MINER * this_mean**2)
            / (N_MINER - 1)
        )
        the_fork_rate = dict_items(
            proptime=block_propagation_time,
            hash_mean=this_mean,
            n=N_MINER,
            std=this_std,
        )

        # this_df_block_propagation_time.loc[N_MINER, HASH_STD]

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
        plt.text(0.0017, N_MINER, f"$N={N_MINER}$", color="k")
        # vertical line at sigma=LOG_NORMAL_SIGMA
        plt.axvline(x=this_std, color="white", linestyle="--", linewidth=0.5)
        plt.text(this_std, 7, f"$s={round(this_std,5)}$", color="k")

        # add dot at n=19 and sigma=LOG_NORMAL_SIGMA
        plt.plot(this_std, N_MINER, "*", color="red", markersize=10)

        plt.xscale("log")
        # plt.yscale("log")
        # log y with base 2
        plt.yscale("log", base=2)
        # plt.xticks(x_ticks, [f"{tick:.1f}" for tick in x_ticks])
        cbar = plt.colorbar(label="fork rate $C(\Delta_0)$", location="top")
        # make cbar ticker labels scientific
        cbar.formatter.set_powerlimits((0, 0))

        # add a shaded region in the horizontal colorbar to indicate the 90% confidence interval, shade patter "////"
        cbar.ax.fill_betweenx(
            [0, 1],
            p5,
            p95,
            color="grey",
            alpha=0.8,
            linewidth=1,
            edgecolor="black",
            hatch="////",
        )

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
            / f"fork_rate_heatmap_{round(block_propagation_time,4)}_{round(SUM_HASH_RATE,4)}_{key}.pdf",
            bbox_inches="tight",
        )
        plt.show()
