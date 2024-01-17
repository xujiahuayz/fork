import json

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER

# load rates from json
with open(DATA_FOLDER / "rates1e6.json", "r") as f:
    rates = json.load(f)
# for each block propagation time, plot the fork rate as y-axis and n as x-axis for each distribution as line type
df = pd.DataFrame(rates)

# export df to excel
# df.to_excel(DATA_FOLDER / "rates1e7.xlsx")
for block_propagation_time in df["block_propagation_time"].unique():
    df_block_propagation_time = df[
        df["block_propagation_time"] == block_propagation_time
    ]
    for distribution in df_block_propagation_time["distribution"].unique():
        df_distribution = df_block_propagation_time[
            df_block_propagation_time["distribution"] == distribution
        ]
        plt.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=f"{distribution}, {block_propagation_time}",
        )
        # set y axis limits
        # plt.ylim(0, 0.018)
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("fork rate")
    plt.title("fork rate vs n")
    plt.savefig(FIGURES_FOLDER / f"fork_rate_vs_n_{block_propagation_time}.pdf")
    plt.show()
