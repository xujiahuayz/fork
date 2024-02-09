import json
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER

# Load rates from json
with open(DATA_FOLDER / "rates_integ_log_normal.json", "r") as f:
    rates = json.load(f)

df = pd.DataFrame(rates)

with open(DATA_FOLDER / "rates_integ_log_normal_add.json", "r") as f:
    rates_add = json.load(f)

df_add = pd.DataFrame(rates_add)
df = pd.concat([df, df_add])

plt.rcParams.update({"font.size": 20})

for block_propagation_time in [0.86, 8.7, 1000]:
    for n_miner in [2, 5, 19, 30]:
        df_distribution = df[
            (df["block_propagation_time"] == block_propagation_time)
            & (df["n"] == n_miner)
            & (df["sigma"] > 0.5)
        ]
        # sort by sigma
        df_distribution = df_distribution.sort_values(by="sigma")

        plt.plot(
            df_distribution["sigma"],
            df_distribution["rate"],
            label=n_miner,
            alpha=0.8,
            linewidth=2,
        )

    plt.legend(
        title="$n$",
        frameon=False,
        loc="upper right",
    )

    plt.xlabel("$\sigma$")
    plt.ylabel("fork rate $C(\Delta_0)$")

    # # Use scientific notation on y-axis and position the exponent on top
    # plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # plt.gca().yaxis.get_offset_text().set_position((0, 1))
    # plt.gca().yaxis.get_offset_text().set_verticalalignment("bottom")

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_sigma_{block_propagation_time}.pdf",
        bbox_inches="tight",
    )
    plt.show()
