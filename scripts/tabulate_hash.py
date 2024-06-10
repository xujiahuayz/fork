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


manipulate_precision("average_block_time", 2)
manipulate_precision("total_hash_rate", 5)
manipulate_precision("max_share", 2)
manipulate_precision("exp_rate", 0)
manipulate_precision("log_normal_loc", 2)
manipulate_precision("log_normal_sigma", 2)
manipulate_precision("lomax_c", 2)
manipulate_precision("lomax_scale", 6)
manipulate_precision("hash_mean", 6)
manipulate_precision("num_miners", 0)
manipulate_precision("hash_std", 6)


hash_panel_to_latex = hash_panel[
    [
        "start_block",
        "start_time",
        "average_block_time",
        "total_hash_rate",
        "num_miners",
        "hash_mean",
        "hash_std",
        "max_share",
        "exp_rate",
        "log_normal_loc",
        "log_normal_sigma",
        "lomax_c",
        "lomax_scale",
    ]
].to_latex(index=False, header=False)

# save to file in table folder
with open(TABLE_FOLDER / "hash_panel.tex", "w") as f:
    f.write(hash_panel_to_latex)
