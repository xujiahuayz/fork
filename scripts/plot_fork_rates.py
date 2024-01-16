import json

import pandas as pd
from matplotlib import pyplot as plt

from fork_env.constants import DATA_FOLDER

# load rates from json
with open(DATA_FOLDER / "rates.json", "r") as f:
    rates = json.load(f)
# for each block propagation time, plot the fork rate as y-axis and n as x-axis for each distribution as line type
df = pd.DataFrame(rates)
for block_propagation_time in df["block_propagation_time"].unique():
    fig, ax = plt.subplots(figsize=(10, 10))
    df_block_propagation_time = df[
        df["block_propagation_time"] == block_propagation_time
    ]
    for distribution in df_block_propagation_time["distribution"].unique():
        df_distribution = df_block_propagation_time[
            df_block_propagation_time["distribution"] == distribution
        ]
        ax.plot(
            df_distribution["n"],
            df_distribution["rate"],
            label=f"{distribution}, {block_propagation_time}",
        )

    ax.legend()
    ax.set_xlabel("n")
    ax.set_ylabel("fork rate")
    # show plot
    plt.show()
