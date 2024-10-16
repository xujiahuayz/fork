import json
import pickle

import pandas as pd

# from scipy.stats import expon

from fork_env.constants import (
    CLUSTER_PATH,
    DATA_FOLDER,
    # DIST_KEYS,
)

# from fork_env.utils import calc_ex_rate, gen_ln_dist, gen_lmx_dist, gen_truncpl_dist
from scripts.get_clusters import btc_tx_value_df, btc_tx_value_series

fork_df = pd.read_pickle(DATA_FOLDER / "forks.pkl")
fork_df[["miner_main", "miner_orphan", "miner_subsequent"]] = pd.DataFrame.map(
    fork_df[["miner_main", "miner_orphan", "miner_subsequent"]],
    lambda x: x.replace('"', "")
    .replace("Solo CKPool", "CKPool Fee")
    .replace("SlushPool", "Braiins Pool")
    .replace("AntPool", "Antpool"),
)
# set block_number as int and index
fork_df["block_number"] = fork_df["block_number"].astype(int)
# merge btc_tx_value_df with fork_df
merged_df = fork_df[["block_number", "miner_main"]].merge(
    btc_tx_value_df[["block_number", "addresses"]],
    left_on="block_number",
    right_on="block_number",
)

# read pool json file as dict
with open(DATA_FOLDER / "pools.json", "r") as f:
    pool_dict = json.load(f)["payout_addresses"]

pool_dict.update(
    {row["addresses"]: {"name": row["miner_main"]} for _, row in merged_df.iterrows()}
)


with open(CLUSTER_PATH, "rb") as f:
    clusters = pickle.load(f)

address_cluster_dict = {
    address: cluster_index
    for cluster_index, cluster in enumerate(clusters)
    for address in cluster
}


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
# block_time_df["miner_addresses"] = btc_tx_value_series
block_time_df["miner_cluster"] = cluster_miner

# pickle the block_time_df
block_time_df.to_pickle(DATA_FOLDER / "block_time_df.pkl")
