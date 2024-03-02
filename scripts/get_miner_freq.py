import pickle

import pandas as pd
from scipy.stats import expon, lognorm, lomax

from fork_env.constants import CLUSTER_PATH  # , SUM_HASH_RATE
from scripts.get_clusters import btc_tx_value_df, btc_tx_value_series

with open(CLUSTER_PATH, "rb") as f:
    clusters = pickle.load(f)

address_cluster_dict = {
    address: cluster_index
    for cluster_index, cluster in enumerate(clusters)
    for address in cluster
}

# for each block, find out which cluster the miner belongs to
cluster_miner = {
    i: set(
        address_cluster_dict[address] if address in address_cluster_dict else -1
        for address in addresses
    )
    for i, addresses in btc_tx_value_series.items()
}

# check if all blocks's cluster length is 1
sum([len(cluster) != 1 for cluster in cluster_miner.values()])  # 0
# check if any cluster is -1
sum([-1 in cluster for cluster in cluster_miner.values()])  # 0

# de-set the cluster_miner and make a Series
cluster_miner = pd.Series(
    {block: list(cluster)[0] for block, cluster in cluster_miner.items()}
).sort_index()

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

    # fit an exponential distribution to miner_hash
    lam = expon.fit(miner_hash)[0]
    # fit a lognormal distribution to miner_hash
    sigma, lognorm_loc, lognorm_scale = lognorm.fit(miner_hash)
    # fit a lomax distribution to miner_hash
    c, lomax_loc, lomax_scale = lomax.fit(miner_hash)

    num_miners = len(miner_hash)
    hash_mean = miner_hash.mean()
    hash_std = miner_hash.std()

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
        "exp_lambda": lam,
        "log_normal_sigma": sigma,
        "log_normal_loc": lognorm_loc,
        "log_normal_scale": lognorm_scale,
        "lomax_c": c,
        "lomax_loc": lomax_loc,
        "lomax_scale": lomax_scale,
    }
    hash_panel.append(row)


hash_panel = pd.DataFrame(hash_panel)

# hash_panel = hash_panel.append(
#     {
#         "block_range": f"{start_block} - {end_block}",
#         "start_time": start_time,
#         "end_time": end_time,
#         "num_miners": num_miners,
#         "hash_mean": hash_mean,
#         "hash_std": hash_std,
#     },
#     ignore_index=True,
# )

# combine block_time, cluster_miner and btc_tx_value_series into a DataFrame with block_number as index


# blocks_in_scope = cluster_miner.loc[start_block : start_block + rolling_window - 1]
# # get frequency of each miner cluster
#
# print(len(miner_freq))
# # plot the frequency with x-axis label as cluster index, and title as block range
# miner_freq.plot(
#     kind="bar",
#     xlabel="Miner cluster Index",
#     ylabel="Frequency",
#     title=f"Block {start_block} - {start_block + rolling_window - 1}",
# )
