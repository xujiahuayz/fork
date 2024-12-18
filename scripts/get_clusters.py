import gzip
import pickle

import pandas as pd

from fork_env.constants import BITCOIN_MINER_PATH, CLUSTER_PATH

with gzip.open(BITCOIN_MINER_PATH, "rb") as f:
    btc_tx_value = pickle.load(f)

# make dataframes from the dict
btc_tx_value_df = pd.DataFrame(btc_tx_value)

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

    class UnionFind:
        def __init__(self):
            self.parent = {}
            self.rank = {}

        def find(self, item):
            # If the item is not in the parent map, initialize it as its own parent
            if item not in self.parent:
                self.parent[item] = item
                self.rank[item] = 0
            # Path compression
            if self.parent[item] != item:
                self.parent[item] = self.find(self.parent[item])
            return self.parent[item]

        def union(self, item1, item2):
            root1 = self.find(item1)
            root2 = self.find(item2)

            if root1 != root2:
                # Union by rank
                if self.rank[root1] > self.rank[root2]:
                    self.parent[root2] = root1
                elif self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                else:
                    self.parent[root2] = root1
                    self.rank[root1] += 1

    # Convert each set to a frozenset for deduplication
    unique_frozensets = set(frozenset(s) for s in btc_tx_value_series)

    # # Deduplicate by converting the list of frozensets to a set
    #  = frozenset_list)

    # Union-Find structure to track clusters
    uf = UnionFind()

    # Go through each unique set of addresses
    for addresses in unique_frozensets:
        addresses = list(addresses)  # Convert frozenset to list for iteration
        for i in range(1, len(addresses)):
            # Union the first element with every other element in the set
            uf.union(addresses[0], addresses[i])

    # Grouping addresses by their root
    clusters_dict = {}
    for addresses in unique_frozensets:
        root = uf.find(list(addresses)[0])  # Find the root of the set
        if root not in clusters_dict:
            clusters_dict[root] = set()
        clusters_dict[root].update(addresses)

    # Convert the clusters dict back into a list of sets
    clusters = list(clusters_dict.values())

    #     return clusters

    # # Usage example
    # # btc_tx_value_series = [{"a", "b"}, {"a"}, {"b", "c"}, {"d"}, {"a", "c"}, {"d", "f"}]

    # clusters = cluster_addresses(btc_tx_value_series)

    # save clusters to a pickle file
    with gzip.open(CLUSTER_PATH, "wb") as f:
        pickle.dump(clusters, f)
