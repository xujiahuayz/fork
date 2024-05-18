import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon, lognorm, lomax

from fork_env.constants import CLUSTER_PATH, DATA_FOLDER, FIGURES_FOLDER
from fork_env.utils import (
    calc_ex_rate,
    calc_ln_loc,
    calc_ln_scale,
    calc_ln_sig,
    calc_lmx_shape,
    calc_lmx_scale,
)
from scripts.get_clusters import btc_tx_value_df, btc_tx_value_series

with open(CLUSTER_PATH, "rb") as f:
    clusters = pickle.load(f)

address_cluster_dict = {
    address: cluster_index
    for cluster_index, cluster in enumerate(clusters)
    for address in cluster
}

# read pool json file as dict
with open(DATA_FOLDER / "pools.json", "r") as f:
    pool_dict = json.load(f)["payout_addresses"]


# for each block, find out which cluster the miner belongs to
cluster_miner_full = {
    i: [
        set(
            address_cluster_dict[address] if address in address_cluster_dict else -1
            for address in addresses
        ),
        set(
            pool_dict[address]["name"] if address in pool_dict else -1
            for address in addresses
        ),
    ]
    for i, addresses in btc_tx_value_series.items()
}


# check if all blocks's cluster length is 1
sum([len(cluster[0]) != 1 for cluster in cluster_miner_full.values()])  # 0
# check if any cluster is -1
sum([-1 in cluster[0] for cluster in cluster_miner_full.values()])  # 0


# check if all blocks's cluster length is 1
sum([len(cluster[1]) >= 3 for cluster in cluster_miner_full.values()])  # 0
# find out which cluster has more than 3 pools
cluster_miner_subset = {
    block: cluster
    for block, cluster in cluster_miner_full.items()
    if len(cluster[1]) >= 3
}


# for each block, find out which cluster the miner belongs to
cluster_miner = pd.Series(
    {
        i: (list(cluster[1] - {-1}) + list(cluster[0] - {-1}))[0]
        for i, cluster in cluster_miner_full.items()
    }
)

block_time_df = (
    btc_tx_value_df.groupby("block_number")["block_timestamp"]
    .first()
    .to_frame(name="block_timestamp")
)
block_time_df["miner_addresses"] = btc_tx_value_series
block_time_df["miner_cluster"] = cluster_miner


first_start_block = 530_000
rolling_window = 30_000

hash_panel = []

for start_block in range(
    first_start_block,
    max(btc_tx_value_series.index) - rolling_window + 1,
    rolling_window,
):
    end_block = start_block + rolling_window

    df_in_scope = block_time_df.loc[start_block:end_block]
    block_times = df_in_scope["block_timestamp"].to_list()
    # time difference between the first and last block in seconds
    start_time = block_times[0]
    end_time = block_times[-1]
    # use miliseconds for time difference
    average_block_time = (end_time - start_time).total_seconds() / rolling_window
    total_hash_rate = 1 / average_block_time
    miner_hash_share = df_in_scope["miner_cluster"].value_counts(normalize=True)

    max_share = miner_hash_share.max()
    min_share = miner_hash_share.min()

    miner_hash = miner_hash_share * total_hash_rate

    num_miners = len(miner_hash)
    hash_mean = miner_hash.mean()
    hash_std = miner_hash.std()

    expon_rate = calc_ex_rate(hash_mean)
    expon_dist = expon(scale=hash_mean)

    # fit a lognormal distribution to miner_hash using moments
    lognorm_sigma = calc_ln_sig(hash_mean, hash_std)
    lognorm_loc = calc_ln_loc(
        hash_mean, hash_std
    )  # "mu", mean of the log of the distribution, not the mean of the distribution
    lognorm_scale = calc_ln_scale(hash_mean, hash_std)
    lognormal_dist = lognorm(lognorm_sigma, scale=lognorm_scale)

    # fit a lomax distribution  using moments
    lomax_shape = calc_lmx_shape(hash_mean, hash_std)
    lomax_scale = calc_lmx_scale(hash_mean, hash_std)
    lomax_dist = lomax(lomax_shape, scale=lomax_scale)

    # add a row to hash_panel
    row = {
        "start_block": start_block,
        "end_block": end_block,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "average_block_time": average_block_time,
        "total_hash_rate": total_hash_rate,
        "num_miners": num_miners,
        "hash_mean": hash_mean,
        "hash_std": hash_std,
        "max_share": max_share * 100,
        "exp_rate": expon_rate,
        "log_normal_loc": lognorm_loc,
        "log_normal_sigma": lognorm_sigma,
        "lomax_c": lomax_shape,
        "lomax_scale": lomax_scale,
    }
    hash_panel.append(row)

    # plot ccdf

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(3, 2))

    # plotting empirical ccdf
    n, bins, patches = ax.hist(
        miner_hash,
        bins=200,
        alpha=0.5,
        label="empirical hash rate",
        cumulative=-1,
        density=True,
    )

    ax.set_ylabel("ccdf")  # we already handled the x-label with ax1
    ax.set_xlabel("hash rate [s$^{-1}$]")
    # Generate points on the x-axis:
    x = np.exp(np.linspace(min(np.log(miner_hash)), max(np.log(miner_hash)), 100))
    # Plotting the fitted distributions on the ax2 with the 'right' y-axis
    # Exponential
    ax.plot(
        x,
        1 - expon_dist.cdf(x),
        label="exponential fit",
    )
    # Log-Normal
    ax.plot(
        x,
        1 - lognormal_dist.cdf(x),
        label="log-normal fit",
    )

    # Lomax
    ax.plot(
        x,
        1 - lomax_dist.cdf(x),
        label="lomax fit",
    )

    # legend top of the plot, outside of the plot, no frame
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 1.4), frameon=False)

    # fix x-axis and y-axis
    ax.set_xlim(4e-8, 6e-4)
    ax.set_ylim(1e-8, 2)

    # log x-axis and y-axis
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()  # adjust the layout to make room for the second y-label
    # save the plot
    plt.savefig(
        FIGURES_FOLDER / f"hash_dist_{start_block}_{end_block}.pdf",
        bbox_inches="tight",
    )


hash_panel = pd.DataFrame(hash_panel)
# save to pickle
hash_panel.to_pickle(DATA_FOLDER / "hash_panel.pkl")
