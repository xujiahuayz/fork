# read hash_panel.pkl from file
import pandas as pd
from fork_env.constants import DATA_FOLDER, TABLE_FOLDER

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")


# make start time exact to date
hash_panel["start_time"] = pd.to_datetime(hash_panel["start_time"]).dt.date


def manipulate_precision(col_ame: str, precision: int) -> None:
    hash_panel[col_ame] = hash_panel[col_ame].apply(
        lambda x: "$" + f"{{:,.{precision}f}}".format(x) + "$"
    )


for proptime in ["50", "90", "99"]:
    hash_panel[f"proptime_{proptime}"] = [
        w[proptime]["proptime"] for w in hash_panel["block_dict"]
    ]
    manipulate_precision(f"proptime_{proptime}", 2)

hash_panel["stale_rate_100"] = hash_panel["stale_rate"] * 100
manipulate_precision("average_block_time", 1)
manipulate_precision("total_hash_rate", 5)
manipulate_precision("max_share", 2)
manipulate_precision("exp_rate", 0)
manipulate_precision("log_normal_loc", 2)
manipulate_precision("log_normal_sigma", 2)
manipulate_precision("lomax_c", 2)
manipulate_precision("lomax_scale", 6)
manipulate_precision("truncpl_alpha", 2)
manipulate_precision("truncpl_ell", 0)
manipulate_precision("hash_mean", 6)
manipulate_precision("num_miners", 0)
manipulate_precision("hash_std", 6)
manipulate_precision("hash_skew", 2)
manipulate_precision("hash_kurt", 2)
manipulate_precision("avg_logged_difficulty", 2)
manipulate_precision("stale_rate_100", 3)


hash_panel_to_latex = hash_panel[
    [
        "start_block",
        "start_time",
        "proptime_50",
        "proptime_90",
        "proptime_99",
        "average_block_time",
        "avg_logged_difficulty",
        "stale_rate_100",
        "num_miners",
        "total_hash_rate",
        "hash_mean",
        "hash_std",
        "hash_skew",
        "hash_kurt",
        "max_share",
        "exp_rate",
        "log_normal_loc",
        "log_normal_sigma",
        # "lomax_c",
        # "lomax_scale",
        "truncpl_alpha",
        "truncpl_ell",
    ]
].to_latex(index=False, header=False)

# save to file in table folder
with open(TABLE_FOLDER / "hash_panel.tex", "w") as f:
    f.write(hash_panel_to_latex)
