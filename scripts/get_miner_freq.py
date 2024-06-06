import json
import pickle

import pandas as pd
from scipy.stats import expon

from fork_env.constants import (
    CLUSTER_PATH,
    DATA_FOLDER,
    DIST_KEYS,
)
from fork_env.utils import calc_ex_rate, gen_ln_dist, gen_lmx_dist
from fork_env.constants import BLOCK_WINDOW
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

hash_panel = []


for start_block in range(
    first_start_block,
    max(btc_tx_value_series.index) - BLOCK_WINDOW + 1,
    BLOCK_WINDOW,
):
    end_block = start_block + BLOCK_WINDOW

    df_in_scope = block_time_df.loc[start_block:end_block]
    block_times = df_in_scope["block_timestamp"].to_list()
    # time difference between the first and last block in seconds
    start_time = block_times[0]
    end_time = block_times[-1]
    # use miliseconds for time difference
    average_block_time = (end_time - start_time).total_seconds() / BLOCK_WINDOW
    total_hash_rate = 1 / average_block_time
    miner_hash_share_count = df_in_scope["miner_cluster"].value_counts()
    miner_hash_share = miner_hash_share_count / BLOCK_WINDOW

    max_share = miner_hash_share.max()
    min_share = miner_hash_share.min()

    miner_hash = miner_hash_share * total_hash_rate

    num_miners = len(miner_hash)
    hash_mean = miner_hash.mean()
    hash_std = miner_hash.std(ddof=0)

    expon_rate = calc_ex_rate(hash_mean)
    expon_dist = expon(scale=hash_mean)

    # fit a lognormal distribution to miner_hash using moments
    lognorm_loc, lognorm_sigma, lognorm_dist = gen_ln_dist(hash_mean, hash_std)

    # fit a lomax distribution  using moments
    lomax_shape, lomax_scale, lomax_dist = gen_lmx_dist(hash_mean, hash_std)

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
        "miner_hash": miner_hash,
        "distributions": {
            DIST_KEYS[0]: expon_dist,
            DIST_KEYS[1]: lognorm_dist,
            DIST_KEYS[2]: lomax_dist,
        },
    }
    hash_panel.append(row)


hash_panel = pd.DataFrame(hash_panel)
# save to pickle
hash_panel.to_pickle(DATA_FOLDER / "hash_panel.pkl")
