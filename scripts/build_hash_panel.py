import pandas as pd
from scipy.stats import expon
import json

from fork_env.constants import (
    DATA_FOLDER,
)
from fork_env.utils import (
    calc_ex_rate,
    gen_ln_dist,
    gen_lmx_dist,
    gen_truncpl_dist,
)
from fork_env.constants import BLOCK_WINDOW, FIRST_START_BLOCK
from scripts.get_clusters import btc_tx_value_series
from fork_env.integration_exp import fork_rate_exp
from fork_env.integration_ln import fork_rate_ln
from fork_env.integration_lomax import fork_rate_lomax
from fork_env.integration_tpl import fork_rate_tpl
from fork_env.integration_empirical import fork_rate_empirical
from scripts.get_fork import multiplier, final_fork


# unpickle block_time_df
merged_df = pd.read_pickle(DATA_FOLDER / "merged_df.pkl")


# # merge the two dataframes
merged_df["difficulty"] *= 2**32

# timestamp to the nearest date
merged_df["date"] = merged_df["time"] // 86400 * 86400
fork_counts = final_fork[["final_fork"]] * multiplier
# change DatetimeIndex(['2009-01-03', '2009-01-09', '2009-01-12', '2009-01-15' to timestamp
fork_counts["time_stamp"] = pd.to_datetime(fork_counts.index).astype(int) // 10**9

# read hash_rate_mean.json and convert to a dataframe
with open(DATA_FOLDER / "hash_rate_mean.json", "r") as f:
    hash_rate_mean = json.load(f)

hash_rate_mean_df = (
    pd.DataFrame(hash_rate_mean).dropna().rename(columns={"v": "total_hash_rate"})
)

# merge hash_rate_mean_df with block_time_df
merged_df = merged_df.merge(
    hash_rate_mean_df, left_on="date", right_on="t", how="left"
).drop(columns=["t"])


# read invstat.pickle
invstat_df = pd.read_pickle(DATA_FOLDER / "invstat.pkl")

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

    df_in_scope = merged_df.loc[start_block:end_block]
    block_times = df_in_scope["time"].to_list()
    # time difference between the first and last block in seconds
    start_time = block_times[0]
    end_time = block_times[-1]

    # use miliseconds for time difference
    average_block_time = (end_time - start_time) / BLOCK_WINDOW
    total_hash_rate = (
        df_in_scope["total_hash_rate"][:-1].sum() / df_in_scope["difficulty"][:-1].sum()
    )
    # 1 / average_block_time
    miner_hash_share_count = df_in_scope["miner_cluster"][:-1].value_counts()
    miner_hash_share = miner_hash_share_count / BLOCK_WINDOW
    bis = list(miner_hash_share_count.sort_values())
    hhi = sum((b / BLOCK_WINDOW) ** 2 for b in bis)

    # avg_logged_difficulty = df_in_scope["difficulty"].mean()
    # find out fork counts where date is in the range
    fork_rate = (
        fork_counts[
            (fork_counts["time_stamp"] >= start_time)
            & (fork_counts["time_stamp"] <= end_time)
        ]["final_fork"].sum()
        / BLOCK_WINDOW
    )

    # orphan_rate = df_in_scope["orphan_blocks"].sum() / BLOCK_WINDOW
    # stale_rate = df_in_scope["stale_blocks"].sum() / BLOCK_WINDOW

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
        (invstat_df["start_unix"] >= start_time) & (invstat_df["end_unix"] <= end_time)
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
                identical=None,
            ),
        }
        block_dict[percentile] = this_block_dict

    expon_rate = calc_ex_rate(hash_mean)
    expon_dist = expon(scale=hash_mean)

    # fit a lognormal distribution to miner_hash using moments
    lognorm_loc, lognorm_sigma, lognorm_dist = gen_ln_dist(hash_mean, hash_std)

    # fit a lomax distribution  using moments
    lomax_shape, lomax_scale, lomax_dist = gen_lmx_dist(
        hash_mean=hash_mean, hash_std=hash_std
    )

    # # fit a truncated power law distribution using moments
    truncpl_alpha, truncpl_ell, truncpl_dist = gen_truncpl_dist(
        hash_mean=hash_mean, hash_std=hash_std
    )

    # add a row to hash_panel
    row = {
        "start_block": start_block,
        "end_block": end_block,
        "start_time": start_time,
        "end_time": end_time,
        "average_block_time": average_block_time,
        "total_hash_rate": total_hash_rate,
        "fork_rate": fork_rate * 100,
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
        "miner_hash": miner_hash,
        "bis": bis,
        "hhi": hhi,
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
