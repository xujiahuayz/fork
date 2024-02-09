import json
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    FIGURES_FOLDER,
    LOG_NORMAL_SIGMA,
    N_MINER,
    SUM_HASH_RATE,
)

# Load rates from json
with open(DATA_FOLDER / "rates_integ_sumhash.json", "r") as f:
    rates = json.load(f)

df = pd.DataFrame(rates)

with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates_integ = json.load(f)

df_integ = pd.DataFrame(rates_integ)
df_integ = df_integ[df_integ["block_propagation_time"] == 8.7]
# remove the distribution column and add sigma column with value = LOG_NORMAL_SIGMA
df_integ = df_integ.drop(columns=["block_propagation_time"])
df_integ["sumhash"] = SUM_HASH_RATE

df = pd.concat([df, df_integ])

plt.rcParams.update({"font.size": 20})

for distribution in ["exp", "log_normal", "lomax"]:
    for sumhash in sorted([1e-3, SUM_HASH_RATE, 1e-1, 1e0], reverse=True):
        df_distribution = df[
            (df["distribution"] == distribution) & (df["sumhash"] == sumhash)
        ]

        df_distribution = df_distribution.sort_values(by=["sumhash", "n"])

        sumhash = round(sumhash, 4)
        plt.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=sumhash,
            alpha=0.8,
            linewidth=2,
        )

    plt.legend(
        title="$\Sigma \lambda$",
        frameon=False,
        loc="right",
        bbox_to_anchor=(1.05, 0.56),
    )

    plt.xlabel("number of miners $n$")
    plt.ylabel("fork rate $C(\Delta_0)$")
    # log y axis
    plt.yscale("log")

    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_hash_{distribution}.pdf",
        bbox_inches="tight",
    )
    plt.show()
