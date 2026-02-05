from scipy.stats import expon
from matplotlib import pyplot as plt
import statsmodels.api as sm  # recommended import according to the docs
from fork_env.constants import (
    DATA_FOLDER,
    # DIST_KEYS,
    FIGURES_FOLDER,
    SUM_HASH_RATE,
    BLOCK_WINDOW,
    DIST_DICT,
)

from fork_env.utils import ccdf_p, gen_ln_dist, gen_truncpl_dist
import pandas as pd
import numpy as np

# import math
from fork_env.utils import calc_ex_rate


hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate
FACTOR = BLOCK_WINDOW / SUM_HASH_RATE
SELECTED_ROW = hash_panel[hash_panel["start_block"] == 868896]
MINER_HASH_EMP = SELECTED_ROW["miner_hash"].iloc[0].sort_values()
BIS = SELECTED_ROW["bis"].iloc[0]


x = [10**i for i in np.linspace(-8, -3, 100)]
EMPFIT_X = [0, 4e-8] + list(MINER_HASH_EMP) + [SUM_HASH_RATE / 3, SUM_HASH_RATE / 2]

for n_zerominers in [20, 100, 200]:
    bis_with_zero_miners = [0] * n_zerominers + BIS
    n = len(bis_with_zero_miners)

    miner_hash: list[float] = list(MINER_HASH_EMP) + [0] * n_zerominers

    hash_mean: float = np.mean(miner_hash)  # type: ignore
    hash_std: float = np.std(miner_hash, ddof=0)  # type: ignore

    expon_rate = calc_ex_rate(hash_mean)
    expon_dist = expon(scale=hash_mean)

    # # fit a lognormal distribution to miner_hash using moments
    lognorm_loc, lognorm_sigma, lognorm_dist = gen_ln_dist(hash_mean, hash_std)

    # fit a lomax distribution  using moments
    # lomax_shape, lomax_scale, lomax_dist = gen_lmx_dist(hash_mean, hash_std)

    truncpl_alpha, truncpl_ell, truncpl_dist = gen_truncpl_dist(
        hash_mean=hash_mean, hash_std=hash_std
    )

    EMPFIT_Y = [
        ccdf_p(
            lbda=lbda,
            bis=bis_with_zero_miners,
            factor=FACTOR,
        )
        for lbda in EMPFIT_X
    ]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(3, 2))

    for key, dist in zip(
        [
            "exp",
            "log_normal",
            # "lomax",
            "trunc_power_law",
        ],
        [
            expon_dist,
            lognorm_dist,
            # lomax_dist,
            truncpl_dist,
        ],
    ):
        ax.plot(
            x,
            dist.sf(x),
            label=DIST_DICT[key]["label"],
            color=DIST_DICT[key]["color"],
        )

    miner_hash.sort()

    ax.plot(
        EMPFIT_X,
        EMPFIT_Y,
        label="semi-empirical, i.i.d.",
        color="black",
        # dashed line
        linestyle="--",
    )
    # plot empirical ccdf as volume plot with steps
    ecdf = sm.distributions.ECDF(miner_hash, side="left")

    ax.fill_between(
        miner_hash,
        (1 - ecdf(miner_hash)).tolist(),
        color="black",
        alpha=0.2,
        label="empirical",
        step="post",
    )

    ax.set_ylabel("ccdf")  # we already handled the x-label with ax1
    ax.set_xlabel("hash rate $\lambda$ [s$^{-1}$]")
    # Generate points on the x-axis:

    # legend top of the plot, outside of the plot, no frame, short legend handles
    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.92, 1.3),
        frameon=False,
        ncol=2,
        fontsize=9,
        handlelength=1,
    )

    # fix x-axis and y-axis
    ax.set_xlim(6e-8, 8e-4)
    ax.set_ylim(2e-3, 1.8)

    # log x-axis and y-axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    # show the plot
    # plt.show()

    fig.tight_layout()  # adjust the layout to make room for the second y-label
    # save the plot
    plt.savefig(
        FIGURES_FOLDER / f"hash_dist_zerominer_{n_zerominers}.pdf",
        bbox_inches="tight",
    )
