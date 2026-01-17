import gzip
import json
import pickle
import re
import numpy as np

import pandas as pd

from fork_env.constants import BLOCK_MINER_CLOVERPOOL_PATH, CLUSTER_PATH, DATA_FOLDER

# from fork_env.utils import calc_ex_rate, gen_ln_dist, gen_lmx_dist, gen_truncpl_dist
from scripts.get_clusters import btc_tx_value_df_exploded, btc_tx_value_series

REPLACEMENTS = [
    (r'"', ""),  # Remove quotes
    (r"^1thash", "58COIN&1THash"),
    (r"antpool", "AntPool"),
    (r"bitclub network", "BitClub"),
    (r"bitcoin\.com.*", "Bitcoin.com"),
    (r"bitcoinindia", "Bitcoin India"),
    (r"btc\.com", "CloverPool"),
    (r"btcc pool", "BTCC"),
    (r"bwpool", "BW.COM"),
    (r"canoepool", "CANOE"),
    (r"ckpool fee", "CKPool"),
    (r"^huobi.*", "Huobi"),
    (r"kucoinpool", "Kucoin"),
    (r"lubian\.com", "Lubian"),
    (r"mara pool", "Mara Pool"),
    (r"patel's mining pool", "Patels"),
    (r"pega pool", "Pega Pool"),
    (r"slushpool", "Braiins Pool"),
    (r"solo ck.*", "CKPool"),  # Match Solo CK* with any suffix
    (r"ultimus pool", "Ultimus"),
    (r"xbtc\.exx\.com&bw\.com;", "EXX&BW"),
    (r"yourbtc\.net", "YourBTC"),
]


def clean_pool_name(pool_name: str) -> str:

    for pattern, replacement in REPLACEMENTS:
        pool_name = re.sub(pattern, replacement, pool_name, flags=re.IGNORECASE)

    return pool_name


fork_df = pd.read_pickle(DATA_FOLDER / "forks.pkl")

fork_df[["miner_main", "miner_orphan", "miner_subsequent"]] = pd.DataFrame.map(
    fork_df[["miner_main", "miner_orphan", "miner_subsequent"]], clean_pool_name
)


fork_df["block_number"] = fork_df["block_number"].astype(int)


cloverpool_df = pd.read_pickle(BLOCK_MINER_CLOVERPOOL_PATH)[["height", "extras"]]

# Name replacement in cloverpool dataframe
cloverpool_df["miner_main"] = cloverpool_df["extras"].apply(
    lambda x: clean_pool_name(x["minerName"])
)

# merge cloverpool_df with fork_df, make sure all fork_df values are there
check_df = fork_df[["block_number", "miner_main"]].merge(
    cloverpool_df[["height", "miner_main"]],
    left_on="block_number",
    right_on="height",
    how="left",
)
# find all miner_main_x is unequal to miner_main_y
assert (
    len(check_df[check_df["miner_main_x"] != check_df["miner_main_y"]]) == 0
), "Not all values are in cloverpool_df"  # if true, then no need to use fork_df as all info is in cloverpool_df

# merge cloverpool_df[['height', 'miner_main']] where miner_main is not unknown with btc_tx_value_df_exploded[['block_number', 'addresses']]
address_miner_df = btc_tx_value_df_exploded[['block_number', 'addresses']].merge(
    cloverpool_df[['height', 'miner_main']][cloverpool_df['miner_main'] != 'unknown'],
    left_on="block_number",
    right_on="height",
    how="inner",
)

# make dict out of address_miner_df; later rows replace earlier rows if there are duplicate addresses
pool_dict = {address: miner for address, miner in zip(address_miner_df['addresses'], address_miner_df['miner_main'])}

# read pool json file as dict
with open(DATA_FOLDER / "pools.json", "r") as f:
    address_miner = json.load(f)["payout_addresses"]

address_miner = {key: clean_pool_name(value["name"]) for key, value in address_miner.items()}
# update pool_dict with address_miner_dict
pool_dict.update(address_miner)


with gzip.open(CLUSTER_PATH, "rb") as f:
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
            pool_dict[address] if address in pool_dict else -1 for address in addresses
        ),
    ]
    for i, addresses in btc_tx_value_series.items()
}


# check if all blocks's cluster length is 1
sum([len(cluster[0]) != 1 for cluster in cluster_miner_full.values()])  # 0
# check if any cluster is -1
sum([-1 in cluster[0] for cluster in cluster_miner_full.values()])  # 0


# check if all blocks's cluster length is 1
sum([len(cluster[1]) >= 3 for cluster in cluster_miner_full.values()]) 

# find out which cluster has more than 3 pools
cluster_miner_subset = {
    block: cluster
    for block, cluster in cluster_miner_full.items()
    if len(cluster[1]) >= 3
} # 2586 - not too bad



# for each block, find out which cluster the miner belongs to
cluster_miner = pd.Series(
    {
        i: (list(cluster[1] - {-1}) + list(cluster[0] - {-1}) or [None])[0]
        for i, cluster in cluster_miner_full.items()
    }
)


# find unique rows in btc_tx_value_df_exploded[['block_number', 'block_timestamp', 'bits']]
merged_df = btc_tx_value_df_exploded[['block_number', 'block_timestamp', 'bits']].drop_duplicates().reset_index(drop=True)
merged_df["miner"] = cluster_miner

merged_df = merged_df.merge(
    cloverpool_df[["height", "miner_main"]],
    left_on="block_number",
    right_on="height",
    how="left",
)
# "miner_main" missing values are replaced with "unknown"
merged_df["miner_main"] = merged_df["miner_main"].fillna("unknown")

# for each row, if miner_main is "unknown", replace it with miner_cluster
merged_df["miner_cluster"] = merged_df.apply(
    lambda row: (row["miner"] if row["miner_main"] == "unknown"  else row["miner_main"]),
    axis=1,
)
# find out rows where merged_df['miner'] is not equal to merged_df['miner_cluster']
merged_df_check = merged_df[merged_df["miner"] != merged_df["miner_cluster"]]


# check if miner_main is not unkown, and miner is not digit, whether the two values are the same, else return true
merged_df_check = merged_df[
    (merged_df["miner_main"] != "unknown")
    & (merged_df["miner"].apply(lambda x: not str(x).isdigit()))
    & (merged_df["miner_main"] != merged_df["miner"])
]
# check merged_df['difficulty']* (2**32) equivalent to merged_df['bits'].apply(bits_to_difficulty)

merged_df[['block_number', 'block_timestamp', 'bits', "miner_cluster"]].to_pickle(
    DATA_FOLDER / "merged_df.pkl"
)
