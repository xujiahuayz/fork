import json

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    FIGURES_FOLDER,
    SUM_HASH_RATE,
)


with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates_integ = json.load(f)

df = pd.DataFrame(rates_integ)
df = df[df["block_propagation_time"] == 8.7]

plt.rcParams.update({"font.size": 20})

for distribution in ["exp", "log_normal", "lomax"]:
    for sumhash in sorted([1e-3, 5e-2, 1, SUM_HASH_RATE], reverse=True):
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
        loc="lower right",
        bbox_to_anchor=(1, -0.05),
        ncol=2,
        handlelength=1,
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
