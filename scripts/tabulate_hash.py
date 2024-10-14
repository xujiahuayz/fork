# read hash_panel.pkl from file
import pandas as pd
from fork_env.constants import DATA_FOLDER, TABLE_FOLDER, VAR_HEADER_UNIT_MAP

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# make start time exact to date
hash_panel["start_time"] = pd.to_datetime(hash_panel["start_time"]).dt.date


def manipulate_precision(col_ame: str, precision: int) -> None:
    hash_panel[col_ame] = hash_panel[col_ame].apply(
        lambda x: f"{{:,.{precision}f}}".format(x)
    )


def conditional_formatting(col_name: str, color: str) -> None:
    hash_panel[col_name] = hash_panel[col_name].apply(
        lambda x: f"\\databar{color}{{{x}}}"
    )


for proptime in ["50", "90", "99"]:
    hash_panel[f"proptime_{proptime}"] = [
        w[proptime]["proptime"] for w in hash_panel["block_dict"]
    ]

hash_panel["stale_rate_100"] = hash_panel["stale_rate"] * 100

for var, header_unit in VAR_HEADER_UNIT_MAP.items():
    if header_unit["precision"] is not None:
        manipulate_precision(var, header_unit["precision"])

# add conditional formatting
conditional_formatting("proptime_50", "red")
conditional_formatting("stale_rate_100", "blue")
conditional_formatting("num_miners", "purple")
conditional_formatting("hash_mean", "orange")
conditional_formatting("hash_std", "orange")
conditional_formatting("max_share", "brown")


hash_panel_to_latex_subtab_1 = hash_panel[
    [k for k, v in VAR_HEADER_UNIT_MAP.items() if v["subtab"] == 1]
].to_latex(index=False, header=False)

hash_panel_to_latex_subtab_2 = hash_panel[
    [k for k, v in VAR_HEADER_UNIT_MAP.items() if v["subtab"] == 2]
].to_latex(index=False, header=False)

# first subtable
with open(TABLE_FOLDER / "hash_empirical.tex", "w", encoding="utf-8") as f:
    f.write(r"\begin{tabular}{@{}cccccccccccrrrr@{}}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(
        r"\multicolumn{2}{c}{period of blocks}"
        + r"  &  "
        + r"\multicolumn{3}{c}{propagation time}"
        + r" & & & & &  "
        + r"\multicolumn{6}{c}{empirical miner hash rate}"
        + r"\\"
        + "\n"
    )
    f.write(r"\cmidrule(lr){1-2} \cmidrule(lr){3-5} \cmidrule(lr){10-15}" + "\n")

    for row in ["header", "unit"]:
        f.write(
            r" & ".join(
                [v[row] for k, v in VAR_HEADER_UNIT_MAP.items() if v["subtab"] == 1]
            )
            + r"\\"
            + "\n"
        )

    f.write(
        r"\cmidrule(lr){1-2} \cmidrule(lr){3-5} \cmidrule(lr){6-9} \cmidrule(lr){10-15}"
        + "\n"
    )
    f.write("\n".join(hash_panel_to_latex_subtab_1.split("\n")[3:]))


# second subtable
with open(TABLE_FOLDER / "hash_dis.tex", "w", encoding="utf-8") as f:
    f.write(r"\begin{tabular}{@{}rrrrr@{}}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"\multicolumn{5}{c}{fitted distributions}" + r"\\" + "\n")
    f.write(r"\cmidrule(lr){1-5}" + "\n")
    f.write(
        r"$\text{Exp}(r)$"
        + r" & "
        + r"\multicolumn{2}{c}{$\text{LN}(\mu, \sigma^2)$}"
        + r" & "
        + r"\multicolumn{2}{c}{$\text{TPL}(\alpha, \beta)$}"
        + r"\\"
        + "\n"
    )
    f.write(r"\cmidrule(lr){1-5}" + "\n")

    for row in ["unit"]:
        f.write(
            r" & ".join(
                [v[row] for k, v in VAR_HEADER_UNIT_MAP.items() if v["subtab"] == 2]
            )
            + r"\\"
            + "\n"
        )
    f.write(r"\cmidrule(lr){1-1} \cmidrule(lr){2-3} \cmidrule(lr){4-5}" + "\n")
    f.write("\n".join(hash_panel_to_latex_subtab_2.split("\n")[3:]))
