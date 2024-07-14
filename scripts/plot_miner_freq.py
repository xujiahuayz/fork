# read hash_panel.pkl from file
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from fork_env.constants import (
    DATA_FOLDER,
    DIST_COLORS,
    DIST_KEYS,
    DIST_LABELS,
    FIGURES_FOLDER,
)
from fork_env.utils import ccdf_p
import statsmodels.api as sm  # recommended import according to the docs


hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

x = [10**i for i in np.linspace(-8, -3, 100)]


# iterate each row

for index, row in hash_panel.iterrows():
    # plot ccdf

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(3, 2))
    bi_hash = row["miner_hash"]

    for key in DIST_KEYS:
        ax.plot(
            x,
            1 - row["distributions"][key].cdf(x),
            label=DIST_LABELS[key],
            color=DIST_COLORS[key],
        )

    emp_x = bi_hash.sort_values().tolist()
    ecdf = sm.distributions.ECDF(emp_x, side="left")
    empfit_x = [0] + emp_x + [row["total_hash_rate"] / 3, row["total_hash_rate"] / 2]
    bis = row["bis"]
    ax.plot(
        empfit_x,
        [
            ccdf_p(
                lbda=lbda,
                bis=bis,
                factor=sum(bis) / row["total_hash_rate"],
            )
            for lbda in empfit_x
        ],
        label="semi-empirical",
        color="red",
        linestyle="--",
    )

    # plot empirical ccdf as volume plot with steps
    ax.fill_between(
        [0] + emp_x,
        [1] + (1 - ecdf(emp_x)).tolist(),
        color="black",
        alpha=0.2,
        label="$\{c \cdot b_i \}$ distribution",
        step="post",
    )

    ax.set_ylabel("ccdf")  # we already handled the x-label with ax1
    ax.set_xlabel("hash rate $\lambda$ [s$^{-1}$]")
    # Generate points on the x-axis:

    # legend top of the plot, outside of the plot, no frame, short legend handles
    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.95, 1.3),
        frameon=False,
        ncol=2,
        fontsize="small",
        handlelength=0.5,
    )

    # fix x-axis and y-axis
    ax.set_xlim(4e-8, 6e-4)
    ax.set_ylim(2e-3, 1.8)

    # log x-axis and y-axis
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()  # adjust the layout to make room for the second y-label
    # save the plot
    plt.savefig(
        FIGURES_FOLDER / f"hash_dist_{index}.pdf",
        bbox_inches="tight",
    )
