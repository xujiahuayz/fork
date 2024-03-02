import pickle

import pandas as pd

from fork_env.constants import BITCOIN_MINER_JSON_PATH, CLUSTER_PATH

# Directly read the JSON lines file into a DataFrame
btc_tx_value_df = pd.read_json(BITCOIN_MINER_JSON_PATH, lines=True)


# Ensure block_number is integer and block_timestamp is datetime
btc_tx_value_df["block_number"] = btc_tx_value_df["block_number"].astype(int)
btc_tx_value_df["block_timestamp"] = pd.to_datetime(btc_tx_value_df["block_timestamp"])

# Explode addresses to separate rows for grouping
btc_tx_value_df = btc_tx_value_df.explode("addresses")

# # Filter out addresses that start with 'nonstandard'
# btc_tx_value_df = btc_tx_value_df[
#     ~btc_tx_value_df["addresses"].str.startswith("nonstandard")
# ]

# Group by block_number, convert addresses to set to eliminate duplicates within a block
btc_tx_value_series = btc_tx_value_df.groupby("block_number")["addresses"].agg(set)

if __name__ == "__main__":
    # Convert each set to a frozenset for deduplication
    frozenset_list = [frozenset(s) for s in btc_tx_value_series]

    # Deduplicate by converting the list of frozensets to a set
    unique_frozensets = set(frozenset_list)

    unique_sets = [set(fs) for fs in unique_frozensets]

    clusters = []

    # unique_sets = [{"a", "b"}, {"a"}, {"b", "c"}, {"d"}, {"a", "c"}, {"d", "f"}]

    for i, addresses in enumerate(unique_sets):
        if i % 1000 == 0:
            print(i)
        # print(addresses)
        # check if any cluster contains any of the addresses
        cluster_index = [
            i for i, cluster in enumerate(clusters) if cluster.intersection(addresses)
        ]
        if len(cluster_index) == 0:
            # if no cluster contains any of the addresses, create a new cluster
            clusters.append(addresses)
        else:
            # if a cluster contains any of the addresses, merge the addresses and the clusters
            cluster = set().union(*[clusters[i] for i in cluster_index])
            cluster.update(addresses)
            # remove the clusters that have been merged
            clusters = [
                cluster for i, cluster in enumerate(clusters) if i not in cluster_index
            ]
            # add the merged cluster
            clusters.append(cluster)

    # save clusters to a pickle file
    with open(CLUSTER_PATH, "wb") as f:
        pickle.dump(clusters, f)
