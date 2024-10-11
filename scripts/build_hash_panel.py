import numpy as np
import pandas as pd
from scipy.stats import expon
import pickle
import json

from fork_env.constants import (
    DATA_FOLDER,
    # DIST_KEYS,
)
from fork_env.utils import calc_ex_rate, gen_ln_dist, gen_lmx_dist, gen_truncpl_dist
from fork_env.constants import BLOCK_WINDOW, FIRST_START_BLOCK
from scripts.get_clusters import btc_tx_value_series
from fork_env.integration_exp import fork_rate_exp
from fork_env.integration_ln import fork_rate_ln
from fork_env.integration_lomax import fork_rate_lomax
from fork_env.integration_tpl import fork_rate_tpl
from fork_env.integration_empirical import fork_rate_empirical


def bits_to_difficulty(bits_hex_str: str) -> float:
    # Ensure the input is 8 characters long
    if len(bits_hex_str) != 8:
        raise ValueError("Invalid bits string. It should be exactly 8 characters long.")

    # Convert the first two characters to the exponent
    exponent = int(bits_hex_str[:2], 16)  # First byte is the exponent
    # Convert the next six characters to the coefficient
    coefficient = int(bits_hex_str[2:], 16)  # Next 6 characters are the coefficient

    # Calculate the target value
    target = coefficient * (256 ** (exponent - 3))

    # Maximum possible target (for difficulty 1)
    max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000

    # Difficulty is the ratio of max_target to target
    return max_target / target


with open(DATA_FOLDER / "btc_tx_difficulty.pkl", "rb") as f:
    difficulty_dict = pickle.load(f)

# create a dataframe from the dict with number as index
difficulty_df = pd.DataFrame(difficulty_dict).set_index("number")
# convert the bits to difficulty, logged
difficulty_df["difficulty"] = (
    difficulty_df["bits"].apply(bits_to_difficulty).apply(np.log)
)


# unpickle block_time_df
block_time_df = pd.read_pickle(DATA_FOLDER / "block_time_df.pkl")

# read orphans.json from data folder
with open(DATA_FOLDER / "n-orphaned-blocks.json", "r") as f:
    orphans = json.load(f)["values"]

orphans_df = pd.DataFrame(orphans)

# read stale blocks from stale-blocks.csv from data folder
stale_blocks = pd.read_csv(DATA_FOLDER / "stale-blocks.csv", header=None).set_index(0)
# group by the block number and count the number of stale blocks
stale_blocks = stale_blocks[[1]].groupby(0).count()
# rename the column to stale_blocks
stale_blocks.columns = ["stale_blocks"]

# merge the two dataframes
block_time_df = block_time_df.merge(
    difficulty_df, left_index=True, right_index=True
).merge(stale_blocks, left_index=True, right_index=True, how="left")
block_time_df["date"] = (
    pd.to_datetime(block_time_df["block_timestamp"].dt.date).astype(int) // 10**9
)

# merge with orphans_df

block_time_df = block_time_df.merge(
    orphans_df, left_on="date", right_on="x", how="left"
).rename(columns={"y": "orphan_blocks"})


# read invstat.gpd

with open(DATA_FOLDER / "invstat.gpd", "rb") as f:
    # the file looks like:
    """
        # Format:
    # $1 + $2: unix timestamp (in milliseconds) of beginning and end of considered interval (usually 1h)
    # $3: total number of INV entries received during period
    # $5: 50% TX propagation delay (milliseconds)
    # $7: 90% TX propagation delay (milliseconds)
    # $9: 99% TX propagation delay (milliseconds)
    # $11: 50% Block propagation delay (milliseconds)
    # $13: 90% Block propagation delay (milliseconds)
    # $15: 99% Block propagation delay (milliseconds)

    1436956763301	1436960364321	28679391	0.5	2276	0.9	10471	0.99	24813	0.5	10507	0.9	21045	0.99	28606	2015-07-15 12:39:23_301	2015-07-15 13:39:24_321
    1437043187210	1437046788257	21788349	0.5	2327	0.9	10772	0.99	24814	0.5	14045	0.9	27359	0.99	29680	2015-07-16 12:39:47_210	2015-07-16 13:39:48_257
    1437129613416	1437133214219	22298279	0.5	2381	0.9	10828	0.99	24881	0.5	10067	0.9	20886	0.99	29003	2015-07-17 12:40:13_416	2015-07-17 13:40:14_219
    1437216037214	1437219638558	16566256	0.5	2344	0.9	10960	0.99	24986	0.5	15312	0.9	24103	0.99	29316	2015-07-18 12:40:37_214	2015-07-18 13:40:38_558
    1437302461328	1437306062095	21521360	0.5	3462	0.9	15645	0.99	27625	0.5	5707	0.9	15789	0.99	27146	2015-07-19 12:41:01_328	2015-07-19 13:41:02_95
    """
    invstat = f.readlines()[11:]

# remove first few lines until the main table starts


# convert the invstat to a dataframe
invstat_df = pd.DataFrame(
    [line.decode("utf-8").split("\t") for line in invstat],
    columns=[
        "start_unix",
        "end_unix",
        "total_inv",
        "tx50_title",
        "tx50",
        "tx90_title",
        "tx90",
        "tx99_title",
        "tx99",
        "block50_title",
        "block50",
        "block90_title",
        "block90",
        "block99_title",
        "block99",
        "start_time",
        "end_time",
    ],
)
# remove rows with missing values
invstat_df = invstat_df.dropna()


# start_unix and end_unix are in milliseconds, convert to seconds
invstat_df["start_unix"] = invstat_df["start_unix"].astype(int) / 1000
invstat_df["end_unix"] = invstat_df["end_unix"].astype(int) / 1000
for proptime in [50, 90, 99]:
    invstat_df[f"block{proptime}"] = invstat_df[f"block{proptime}"].astype(float) / 1000


hash_panel = []


for start_block in range(
    FIRST_START_BLOCK,
    max(btc_tx_value_series.index) - BLOCK_WINDOW + 1,
    BLOCK_WINDOW,
):
    print(start_block)
    end_block = start_block + BLOCK_WINDOW

    df_in_scope = block_time_df.loc[start_block:end_block]
    block_times = df_in_scope["block_timestamp"].to_list()
    # time difference between the first and last block in seconds
    start_time = block_times[0]
    end_time = block_times[-1]

    # use miliseconds for time difference
    average_block_time = (end_time - start_time).total_seconds() / BLOCK_WINDOW
    total_hash_rate = 1 / average_block_time
    miner_hash_share_count = df_in_scope["miner_cluster"][:-1].value_counts()
    miner_hash_share = miner_hash_share_count / BLOCK_WINDOW
    bis = list(miner_hash_share_count.sort_values())

    avg_logged_difficulty = df_in_scope["difficulty"].mean()

    orphan_rate = df_in_scope["orphan_blocks"].sum() / BLOCK_WINDOW
    stale_rate = df_in_scope["stale_blocks"].sum() / BLOCK_WINDOW

    max_share = miner_hash_share.max()
    min_share = miner_hash_share.min()

    miner_hash = miner_hash_share * total_hash_rate

    num_miners = len(miner_hash)
    hash_mean = miner_hash.mean()
    hash_std = miner_hash.std()
    # skewness
    hash_skew = miner_hash.skew()
    # kurtosis
    hash_kurt = miner_hash.kurt()

    # find the chunk of invstat_df that corresponds to the block times
    invstat_chunk = invstat_df[
        (invstat_df["start_unix"] >= start_time.timestamp())
        & (invstat_df["end_unix"] <= end_time.timestamp())
    ]

    block_dict = {}
    for percentile in ["50", "90", "99"]:
        proptime = invstat_chunk[f"block{percentile}"].mean()
        this_block_dict = {
            "proptime": proptime,
            "exp": fork_rate_exp(
                proptime=proptime,
                n=num_miners,
                sum_lambda=total_hash_rate,
                hash_mean=hash_mean,
            ),
            "log_normal": fork_rate_ln(
                proptime=proptime,
                n=num_miners,
                sum_lambda=total_hash_rate,
                hash_mean=hash_mean,
                std=hash_std,
            ),
            "lomax": fork_rate_lomax(
                proptime=proptime,
                n=num_miners,
                sum_lambda=total_hash_rate,
                hash_mean=hash_mean,
                std=hash_std,
            ),
            "trunc_power_law": fork_rate_tpl(
                proptime=proptime,
                n=num_miners,
                sum_lambda=total_hash_rate,
                hash_mean=hash_mean,
                std=hash_std,
            ),
            "empirical": fork_rate_empirical(
                proptime=proptime,
                n=num_miners,
                sum_lambda=total_hash_rate,
                bis=bis,
            ),
        }
        block_dict[percentile] = this_block_dict

    # block50 = invstat_chunk["block50"].mean()
    # block90 = invstat_chunk["block90"].mean()
    # block99 = invstat_chunk["block99"].mean()

    expon_rate = calc_ex_rate(hash_mean)
    expon_dist = expon(scale=hash_mean)

    # fit a lognormal distribution to miner_hash using moments
    lognorm_loc, lognorm_sigma, lognorm_dist = gen_ln_dist(hash_mean, hash_std)

    # fit a lomax distribution  using moments
    lomax_shape, lomax_scale, lomax_dist = gen_lmx_dist(
        hash_mean=hash_mean, hash_std=hash_std
    )

    # # fit a truncated power law distribution using moments
    truncpl_alpha, truncpl_ell, truncpl_scaling_c, truncpl_dist = gen_truncpl_dist(
        hash_mean=hash_mean, hash_std=hash_std
    )

    # add a row to hash_panel
    row = {
        "start_block": start_block,
        "end_block": end_block,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "avg_logged_difficulty": avg_logged_difficulty,
        "orphan_rate": orphan_rate,
        "stale_rate": stale_rate,
        "average_block_time": average_block_time,
        "total_hash_rate": total_hash_rate,
        "num_miners": num_miners,
        "hash_mean": hash_mean,
        "hash_std": hash_std,
        "hash_skew": hash_skew,
        "hash_kurt": hash_kurt,
        "max_share": max_share * 100,
        "exp_rate": expon_rate,
        "log_normal_loc": lognorm_loc,
        "log_normal_sigma": lognorm_sigma,
        "lomax_c": lomax_shape,
        "lomax_scale": lomax_scale,
        "truncpl_alpha": truncpl_alpha,
        "truncpl_ell": truncpl_ell,
        "truncpl_scaling_c": truncpl_scaling_c,
        "miner_hash": miner_hash,
        "bis": bis,
        "block_dict": block_dict,
        "distributions": {
            "exp": expon_dist,
            "log_normal": lognorm_dist,
            "lomax": lomax_dist,
            "trunc_power_law": truncpl_dist,
        },
    }
    hash_panel.append(row)


hash_panel = pd.DataFrame(hash_panel)
# save to pickle
hash_panel.to_pickle(DATA_FOLDER / "hash_panel.pkl")


# hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
# # get the last row of hash panel
# hash_panel_last_row = hash_panel.iloc[-1]

# # get the last number of total hash rate
# SUM_HASH_RATE = hash_panel_last_row["total_hash_rate"]
# N_MINER = hash_panel_last_row["num_miners"]
# HASH_STD = hash_panel_last_row["hash_std"]
