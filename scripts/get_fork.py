import pandas as pd
import json

from fork_env.constants import (
    DATA_FOLDER,
)


# unpickle block_time_df
block_time_df = pd.read_pickle(DATA_FOLDER / "merged_df.pkl")[['block_timestamp', "miner_cluster"]]

fork_df = pd.read_pickle(DATA_FOLDER / "forks.pkl")

fork_df["block_number"] = fork_df["block_number"].astype(int)

# read stale blocks from stale-blocks.csv from data folder
stale_blocks = pd.read_csv(DATA_FOLDER / "stale-blocks.csv")
# group by the block number and count the number of stale blocks
stale_blocks = stale_blocks[['height','hash']].groupby('height').count()
# rename the column to stale_blocks
stale_blocks.columns = ["stale_blocks"]
fork_blocks = fork_df[["block_number", "miner_orphan"]].groupby("block_number").count()
# merge stale_blocks with fork_df
merged_forks_raw = stale_blocks.merge(
    fork_blocks, left_index=True, right_index=True, how="outer"
).sort_index()

# for each row, stale number is the max of the two columns, ignore na
merged_forks_raw["stale_blocks"] = merged_forks_raw["stale_blocks"].fillna(0)
merged_forks_raw["miner_orphan"] = merged_forks_raw["miner_orphan"].fillna(0)
merged_forks_raw["stale_number"] = merged_forks_raw.apply(
    lambda x: max(x["stale_blocks"], x["miner_orphan"]), axis=1
)


# merge the two dataframes
merged_forks = block_time_df[["block_timestamp"]].merge(
    merged_forks_raw, left_index=True, right_index=True, how="left"
)
merged_forks["date_3"] = (
    pd.to_datetime(merged_forks["block_timestamp"].dt.date).astype(int)
    // 10**9
    // 3600
    // 72
)

merged_forks_df = (
    merged_forks.drop(columns=["block_timestamp"]).groupby("date_3").sum().reset_index()
)

with open(DATA_FOLDER / "n-orphaned-blocks.json", "r") as f:
    orphans = json.load(f)["values"]

orphans_df = pd.DataFrame(orphans)
orphans_df["date_3"] = orphans_df["x"] // 3600 // 72

final_fork = merged_forks_df.merge(
    orphans_df, left_on="date_3", right_on="date_3", how="left"
).rename(columns={"y": "orphan_blocks", "x": "date"})

# moving sum of 30 days for all columns, treat na as 0
final_fork["date"] = pd.to_datetime(final_fork["date_3"] * 3600 * 72, unit="s")
# set date as index
final_fork.set_index("date", inplace=True)
final_fork["final_fork"] = final_fork.apply(
    lambda x: max(x["stale_number"], x["orphan_blocks"]), axis=1
)




if __name__ == "__main__":
    # plot line chart for the moving sum of 30 days, with x axis as date
    import matplotlib.pyplot as plt

    N_3day = 71

    final_fork_moving_sum = (
        final_fork.drop(columns=["date_3"])
        .fillna(0)
        .rolling(window=N_3day, center=True)
        .sum()
    )

    N_block = N_3day * 3 * 24 * 60 / 10


    fork_rate = final_fork_moving_sum / N_block * 100

    # fork_rate["date"] = pd.to_datetime(fork_rate.index, unit="s")
    key_date = "2016-02-29"
    # locate value of key date
    multiplier = 0.41 / fork_rate[fork_rate.index == key_date]["final_fork"].values[0]


    # plot time series
    plt.plot(
        fork_rate.index,
        fork_rate[["final_fork", "stale_blocks"]],
    )
    plt.plot(
        fork_rate.index,
        fork_rate["final_fork"] * multiplier,
        # label="orphan_blocks * 0.41",
    )
    # do a vertical line at 2016-02-29
    plt.axvline(x=pd.to_datetime("2016-02-29"), color="r", linestyle="--")
    # x limit to 2015 august to 2024 june
    plt.xlim(pd.to_datetime("2015-07-20"), pd.to_datetime("2024-07-01"))
    plt.ylim(0, 1.19)