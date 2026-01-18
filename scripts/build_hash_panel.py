import pandas as pd
from scipy.stats import expon

from fork_env.constants import BLOCK_WINDOW, DATA_FOLDER, FIRST_START_BLOCK
from fork_env.integration_empirical import fork_rate_empirical
from fork_env.integration_exp import fork_rate_exp
from fork_env.integration_ln import fork_rate_ln, waste_ln
from fork_env.integration_tpl import fork_rate_tpl, waste_tpl
from fork_env.utils import calc_ex_rate, gen_ln_dist, gen_truncpl_dist
from scripts.get_fork import final_fork, multiplier


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
    return max_target / target * (2**32)

# unpickle block_time_df
merged_df = pd.read_pickle(DATA_FOLDER / "merged_df.pkl")
merged_df["date"] = merged_df["block_timestamp"].dt.strftime('%Y-%m-%d')
merged_df['timestamp'] = merged_df['block_timestamp'].astype(int) / 1e9
merged_df["difficulty"] = merged_df["bits"].apply(bits_to_difficulty)

# open btc.csv to get hash
with open(DATA_FOLDER / "btc.csv", "r") as f:
    btc_data = pd.read_csv(f)
btc_data['total_hash_rate'] = btc_data['HashRate'] * 1e12
merged_df = merged_df.merge(
    btc_data[['time', 'total_hash_rate']], left_on="date", right_on="time", how="left"
).drop(columns=['time'])

# read mining_hardware_efficiency.csv and convert to a dataframe
efficiency_df = pd.read_csv(DATA_FOLDER / "mining_hardware_efficiency.csv", skiprows=1).rename(
    columns={"Date": "date", "Estimated efficiency, J/Th": "efficiency"}
)
# merge efficiency_df with block_time_df
merged_df = pd.merge(merged_df, efficiency_df[['date', 'efficiency']], on="date", how="left")

# read invstat.pickle
invstat_df = pd.read_pickle(DATA_FOLDER / "invstat.pkl")

# start_unix and end_unix are in milliseconds, convert to seconds
invstat_df["start_unix"] = invstat_df["start_unix"].astype(int) / 1000
invstat_df["end_unix"] = invstat_df["end_unix"].astype(int) / 1000
for proptime in [50, 90, 99]:
    invstat_df[f"block{proptime}"] = invstat_df[f"block{proptime}"].astype(float) / 1000


fork_counts = (final_fork[["final_fork"]] * multiplier)
fork_counts['date'] = fork_counts.index.astype(int) / 1e9

hash_panel = []

for start_block in range(
    FIRST_START_BLOCK,
    # max(merged_df['block_number']) - BLOCK_WINDOW,
    896_327- BLOCK_WINDOW,
    BLOCK_WINDOW,
):
    print(start_block)
    end_block = start_block + BLOCK_WINDOW

    start_time = merged_df.loc[start_block, "timestamp"]
    end_time = merged_df.loc[end_block, "timestamp"]

    # use miliseconds for time difference
    average_block_time = (end_time - start_time) / BLOCK_WINDOW

    df_in_scope = merged_df.iloc[start_block:end_block]

    total_hash_rate = (
        df_in_scope["total_hash_rate"].sum() / df_in_scope["difficulty"].sum()
    ) # unit: blocks per second

    # 1 / average_block_time
    miner_hash_share_count = df_in_scope["miner_cluster"].value_counts()
    miner_hash_share = miner_hash_share_count / BLOCK_WINDOW
    bis = list(miner_hash_share_count.sort_values())
    hhi = sum((b / BLOCK_WINDOW) ** 2 for b in bis)

    avg_difficulty = df_in_scope["difficulty"].mean()

    avg_efficiency = df_in_scope["efficiency"].mean()

    # find out fork counts where date is in the range
    fork_rate = (
        fork_counts[
            (fork_counts["date"] >= start_time)
            & (fork_counts["date"] <= end_time)
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

    # # fit a truncated power law distribution using moments
    truncpl_alpha, truncpl_ell, truncpl_dist = gen_truncpl_dist(
        hash_mean=hash_mean, hash_std=hash_std
    )

    total_hash_power = (
        total_hash_rate * avg_efficiency * avg_difficulty / (1e12 * 1e6)
    )  # efficiency is in J/THash, convert to MW

    waste_hash_ln = waste_ln(n=num_miners, sum_lambda=total_hash_rate, std=hash_std)

    wasted_power_ln = (
        waste_hash_ln
        * avg_difficulty
        * avg_efficiency
        / 1e12  # efficiency is in J/THash
        / 1e6  # convert to MW
    )

    waste_hash_tpl = waste_tpl(n=num_miners, sum_lambda=total_hash_rate, std=hash_std)

    wasted_power_tpl = (
        waste_hash_tpl
        * avg_difficulty
        * avg_efficiency
        / 1e12  # efficiency is in J/THash
        / 1e6  # convert to MW
    )

    # add a row to hash_panel
    row = {
        "start_block": start_block,
        "end_block": end_block,
        "start_time": start_time * 1e9,
        "end_time": end_time * 1e9,
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
        "truncpl_alpha": truncpl_alpha,
        "truncpl_ell": truncpl_ell,
        "miner_hash": miner_hash,
        "bis": bis,
        "hhi": hhi,
        "block_dict": block_dict,
        "distributions": {
            "exp": expon_dist,
            "log_normal": lognorm_dist,
            "trunc_power_law": truncpl_dist,
        },
        "difficulty": avg_difficulty / 1e12,  # in unit Tera
        "efficiency": avg_efficiency,
        "total_hash_power": total_hash_power,
        "waste_hash_ln": waste_hash_ln,
        "wasted_power_ln": wasted_power_ln,
        "waste_hash_tpl": waste_hash_tpl,
        "wasted_power_tpl": wasted_power_tpl,
    }
    hash_panel.append(row)


hash_panel = pd.DataFrame(hash_panel)
# save to pickle
hash_panel.to_pickle(DATA_FOLDER / "hash_panel.pkl")
