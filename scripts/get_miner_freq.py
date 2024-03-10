import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon, lognorm, lomax

from fork_env.constants import CLUSTER_PATH, DATA_FOLDER
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

# add_cluster_keys = address_cluster_dict.keys()
# pool_keys = pool_dict.keys()
# cluster_miner = {
#     i: [
#         (
#             set(address_cluster_dict[address] for address in add_set)
#             if (add_set := addresses.intersection(add_cluster_keys))
#             else None
#         ),
#         (
#             set(pool_dict[address]["name"] for address in add_set)
#             if (add_set := addresses.intersection(pool_keys))
#             else None
#         ),
#     ]
#     for i, addresses in btc_tx_value_series.items()
# }


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


first_start_block = 455_000
rolling_window = 25_000

hash_panel = []

for start_block in range(
    first_start_block,
    max(btc_tx_value_series.index) - rolling_window + 1,
    rolling_window,
):
    end_block = start_block + rolling_window - 1

    df_in_scope = block_time_df.loc[start_block:end_block]
    block_times = df_in_scope["block_timestamp"].to_list()
    # time difference between the first and last block in seconds
    start_time = block_times[0]
    end_time = block_times[-1]
    average_block_time = (end_time - start_time).total_seconds() / rolling_window
    total_hash_rate = 1 / average_block_time
    miner_hash_share = df_in_scope["miner_cluster"].value_counts(normalize=True)

    max_share = miner_hash_share.max()
    min_share = miner_hash_share.min()

    miner_hash = miner_hash_share * total_hash_rate

    num_miners = len(miner_hash)
    hash_mean = miner_hash.mean()
    hash_std = miner_hash.std()

    # # fit an exponential distribution to miner_hash
    # lam = expon.fit(miner_hash)[0]

    lognorm_sigma, lognorm_loc, lognorm_scale = lognorm.fit(miner_hash)
    # # fit a lomax distribution to miner_hash using moments

    # c = 1 + hash_mean**2 / hash_std**2
    # lomax_loc = hash_mean - hash_mean / c
    # lomax_scale = hash_mean / c
    # # fit a lognormal distribution to miner_hash using moments
    # lognorm_sigma = np.sqrt(np.log(1 + hash_std**2 / hash_mean**2))
    # lognorm_loc = (
    #     np.log(hash_mean) - lognorm_sigma**2 / 2
    # )  # "mu", mean of the log of the distribution, not the mean of the distribution
    # lognorm_scale = np.exp(lognorm_loc)
    # mean of fitted lognormal distribution
    lognorm_mean = lognorm.mean(lognorm_sigma, loc=lognorm_loc, scale=lognorm_scale)
    lognorm_std = lognorm.std(lognorm_sigma, loc=lognorm_loc, scale=lognorm_scale)

    c, lomax_loc, lomax_scale = lomax.fit(miner_hash)
    # mean of fitted lomax distribution
    lomax_mean = lomax.mean(c, loc=lomax_loc, scale=lomax_scale)
    lomax_std = lomax.std(c, loc=lomax_loc, scale=lomax_scale)

    # add a row to hash_panel
    row = {
        "block_range": f"{start_block} - {end_block}",
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "average_block_time": average_block_time,
        "total_hash_rate": total_hash_rate,
        "num_miners": num_miners,
        "hash_mean": hash_mean,
        "hash_std": hash_std,
        "max_share": max_share * 100,
        # "exp_lambda": lam,
        "exp_scale": hash_mean,
        "log_normal_loc": lognorm_loc,
        "log_normal_sigma": lognorm_sigma,
        "log_normal_mean": lognorm_mean,
        "log_normal_std": lognorm_std,
        # "log_normal_scale": lognorm_scale,
        "lomax_c": c,
        "lomax_loc": lomax_loc,
        "lomax_scale": lomax_scale,
        "lomax_mean": lomax_mean,
        "lomax_std": lomax_std,
    }
    hash_panel.append(row)

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting the histogram of miner_hash on the ax1 with the 'left' y-axis
    color = "tab:blue"
    ax1.set_xlabel("Hash Rate")
    ax1.set_ylabel("Frequency", color=color)
    n, bins, patches = ax1.hist(
        miner_hash, bins=200, alpha=0.5, label="Miner Hash Rates", color=color
    )
    ax1.tick_params(axis="y", labelcolor=color)

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel(
        "Probability Density", color=color
    )  # we already handled the x-label with ax1
    ax2.tick_params(axis="y", labelcolor=color)

    # Generate points on the x-axis:
    x = np.linspace(min(miner_hash), max(miner_hash), 100)

    # Plotting the fitted distributions on the ax2 with the 'right' y-axis
    # Exponential
    ax2.plot(
        x,
        expon.pdf(x, scale=hash_mean),
        "r-",
        lw=2,
        label="Exponential Fit (right axis)",
    )
    # Log-Normal
    ax2.plot(
        x,
        lognorm.pdf(x, s=lognorm_sigma, scale=np.exp(lognorm_loc)),
        "g-",
        lw=2,
        label="Log-Normal Fit (right axis)",
    )
    # Lomax
    ax2.plot(
        x,
        lomax.pdf(x, c, loc=lomax_loc, scale=lomax_scale),
        "b-",
        lw=2,
        label="Lomax Fit (right axis)",
    )

    # Adding titles and legend
    plt.title(
        f"Hash Rate Distribution and Fits for Blocks {start_block} to {end_block}"
    )
    fig.tight_layout()  # adjust the layout to make room for the second y-label

    # Create combined legend for both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # Display the plot
    plt.show()
    plt.close()

    # plot miner_hash as histogram with left y-axis, and then plot the fitted distributions on the right y-axis


hash_panel = pd.DataFrame(hash_panel)
