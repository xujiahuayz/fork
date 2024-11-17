import json
import pickle
import csv
import pandas as pd

# from scipy.stats import expon

from fork_env.constants import (
    CLUSTER_PATH,
    DATA_FOLDER,
    BLOCK_MINER_CLOVERPOOL_PATH,
    # DIST_KEYS,
)

# from fork_env.utils import calc_ex_rate, gen_ln_dist, gen_lmx_dist, gen_truncpl_dist
from scripts.get_clusters import btc_tx_value_df, btc_tx_value_series

fork_df = pd.read_pickle(DATA_FOLDER / "forks.pkl")

'''fork_df[["miner_main", "miner_orphan", "miner_subsequent"]] = pd.DataFrame.map(
    fork_df[["miner_main", "miner_orphan", "miner_subsequent"]],
    lambda x: x.replace('"', "")
    .replace("Antpool", "AntPool")
    .replace("Solo CKPool", "CKPool")
    .replace("SlushPool", "Braiins Pool")
    .replace("BTC.com", "CloverPool")
    .replace("BTC.COM", "CloverPool")
    .replace("xbtc.exx.com&bw.com;", "EXX&BW")
    .replace("BTCC", "BTCC Pool")
)'''

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

cluster_miner = cluster_miner.map(
    lambda x: str(x).replace('"', "")
    .replace("BTCC Pool", "BTCC")
    .replace("Antpool", "AntPool")
    .replace("Solo CKPool", "CKPool")
    .replace("SlushPool", "Braiins Pool")
    .replace("BTC.com", "CloverPool")
    .replace("BTC.COM", "CloverPool")
    .replace("xbtc.exx.com&bw.com;", "EXX&BW")
    .replace("Patel's Mining pool", "Patels")
)

block_time_df = (
    btc_tx_value_df.groupby("block_number")["block_timestamp"]
    .first()
    .to_frame(name="block_timestamp")
)
# block_time_df["miner_addresses"] = btc_tx_value_series
block_time_df["miner_cluster"] = cluster_miner

cloverpool_df = pd.read_pickle(BLOCK_MINER_CLOVERPOOL_PATH)

# Name replacement in cloverpool dataframe
cloverpool_df['extras'] = cloverpool_df['extras'].apply(
    lambda d: {
        **d,
        'minerName': d['minerName']
        .replace('"', "")
        .replace("SlushPool", "Braiins Pool")
        .replace("BTC.COM", "CloverPool")
        .replace("BWPool", "BW.COM")
        .replace("Huobi.pool", "Huobi")
        .replace("BitClub", "BitClub Network")
        .replace("MARA Pool", "Mara Pool")
        .replace("Lubian.com", "Lubian")
        .replace("ULTIMUS POOL", "Ultimus")
        .replace("PEGA Pool", "Pega Pool")
        .replace("Yourbtc.net", "YourBTC")
        .replace("CanoePool", "CANOE")
        .replace("BitcoinIndia", "Bitcoin India")
        .replace("Solo CK", "CKPool")
    }
)

cloverpool_df['miner_cluster'] = cloverpool_df['extras'].apply(lambda x: x['minerName'])
# extract needed information only (block height, timestamp and miner_cluster)
clover_block_df = cloverpool_df[['height', 'miner_cluster']]
clover_block_df.rename(columns={'height': 'block_number'}, inplace=True)
clover_block_df.set_index('block_number', inplace=True)

# Merge the two datasets on common columns (e.g., 'block_height')
merged_df = pd.merge(block_time_df, clover_block_df, on=['block_number'], suffixes=('_1', '_2'), how="inner")

# Filter rows where miner_cluster_1 is numeric and miner_cluster_2 is not "unknown"
filtered_df = merged_df[
(merged_df['miner_cluster_1'].apply(lambda x: str(x).isnumeric())) & 
(merged_df['miner_cluster_2'] != "unknown")]

# Get the first occurrence of miner_cluster_2 for each miner_cluster_1
first_occurrences = filtered_df.groupby('miner_cluster_1')['miner_cluster_2'].first()
first_occurrences = first_occurrences.to_dict()

clash_lst = []
any_equivalent = []

def resolve_clashes(row1, row2, index):
    """
    Resolve miner_cluster clashes based on specific rules and conditions.

    Parameters:
    - row1: Row from dataset1 with 'miner_cluster_1'.
    - row2: Row from dataset2 with 'miner_cluster_2'.

    Returns:
    - Resolved miner_cluster value.
    """
    # Case 1: Both miner_cluster values are the same, return the common value.

    if row1['miner_cluster_1'] == row2['miner_cluster_2']:
        any_equivalent.append(index)
        return row1['miner_cluster_1']
    
    # Special Case: Replace '1THash' with '58COIN&1THash' during their pool combination period.
    # Reference: https://remitano.com/cf/forum/5082-10-best-bitcoin-mining-pools-to-join-in-2021
    if row1['miner_cluster_1'] == '1THash' and row2['miner_cluster_2'] == '58COIN&1THash':
        return row2['miner_cluster_2']
    
    # Case 2: If row1 has a known miner_cluster and row2 is unknown, prioritize row1.
    if not str(row1['miner_cluster_1']).isdigit() and row2['miner_cluster_2'] == 'unknown':
        return row1['miner_cluster_1']
    
    # Case 3: If row1 is unknown but row2 has a known miner_cluster, prioritize row2.
    if str(row1['miner_cluster_1']).isdigit() and row2['miner_cluster_2'] != 'unknown':
        return row2['miner_cluster_2']

    # Case 4: If both are known with any clashes, we prioritize row1
    if not str(row1['miner_cluster_1']).isdigit() and row2['miner_cluster_2'] != 'unknown':
        clash_lst.append(index)
        return row1['miner_cluster_1']
    
    # Case 4: Clash scenario where row1 is a cluster number and row2 is unknown.
    # Resolve using the first occurrence of row1's miner_cluster_1 if available.
    if str(row1['miner_cluster_1']).isdigit() and row2['miner_cluster_2'] == 'unknown':
        if row1['miner_cluster_1'] in first_occurrences:
            return first_occurrences[row1['miner_cluster_1']]
        else:
            return row1['miner_cluster_1']  # Default to row1's value if no first occurrence exists.

# Apply the resolve_clashes function row by row across the merged DataFrame.
merged_df['miner_cluster'] = merged_df.apply(
    lambda row: resolve_clashes(row, row, row.name), axis=1
)

# merged_df = merged_df.drop(['miner_cluster_1', 'miner_cluster_2'], axis=1)

value_counts = merged_df["miner_cluster"].value_counts()


# Extract MINER_COUNTRY from Miner_country.csv, and determine top5 largest China pools
MINER_COUNTRY = {}

# Open the CSV file
with open(DATA_FOLDER / "Miner_country.csv", mode="r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["country_based"] != "Unknown":
            MINER_COUNTRY[row["miner_cluster"]] = row["country_based"]

# Extract China pools with their counts
china_pools = {miner: count for miner, count in value_counts.items() if MINER_COUNTRY.get(miner) == "China"}

# Get the top 5 China pools by value counts
top_6_china_pools = set(sorted(china_pools, key=china_pools.get, reverse=True)[:6])

# Replace miners other than the top 5 China pools with "China other"
MINER_COUNTRY = {
    miner: ("China other" if miner not in top_6_china_pools and country == 'China' else country)
    for miner, country in MINER_COUNTRY.items()
}

# merged_df.to_csv(DATA_FOLDER / 'merged_df.csv')
merged_df.to_pickle(DATA_FOLDER / 'merged_df.pkl')
