import pandas as pd
import matplotlib.pyplot as plt

from fork_env.constants import (
    DATA_FOLDER,
    BLOCK_WINDOW,
    FIRST_START_BLOCK,
    FIGURES_FOLDER,
)

# unpickle block_time_df
block_time_df = pd.read_pickle(DATA_FOLDER / "block_time_df.pkl")

# Create a color map for each unique miner_cluster
color_map = {
    miner: color
    for miner, color in zip(
        [
            "Foundry USA",
            "Antpool",
            "F2Pool",
            "ViaBTC",
            "Binance Pool",
            "Mara Pool",
            "Braiins Pool",
            "Poolin",
            "BTC M4",
            "SBI Crypto",
            "BTC.com",
            "BTCC Pool",
            "1Hash",
            "BTC.TOP",
            "BitFury",
            "KnCMiner",
            "BitClub Network",
            "1THash",
            "BTC M6",
        ],
        plt.cm.tab20.colors,
    )
}

for start_block in range(
    FIRST_START_BLOCK,
    870_000,
    BLOCK_WINDOW,
):

    print(start_block)
    end_block = start_block + BLOCK_WINDOW - 1

    df_in_scope = block_time_df.loc[start_block:end_block]

    # Get the value counts for the pie plot and map colors using the color_map
    value_counts = df_in_scope["miner_cluster"].value_counts()

    # Get the corresponding colors from the color_map for each miner in the value_counts
    colors = [
        color_map[miner] if miner in color_map else "gainsboro"
        for miner in value_counts.index
    ]

    # Generate explode values to make pie segments a bit apart from each other
    explode = [0.05] * len(value_counts)

    labels = [
        miner if (miner in color_map) & (value > 400) else ""
        for miner, value in value_counts.items()
    ]

    plt.figure(figsize=(6, 6))

    # plot df_in_scope["miner_cluster"] as pie plot with fixed colors, explode, and custom labels
    value_counts.plot.pie(
        colors=colors,
        labels=labels,  # apply the custom labels here
        startangle=90,
        explode=explode,  # add space between segments
        autopct=lambda pct: (
            f"{pct:.1f}%" if pct > 2 else ""
        ),  # show percentage only for non-zero slices
    )

    # remove the y-axis label
    plt.ylabel("")

    plt.tight_layout()

    # save to file
    plt.savefig(
        FIGURES_FOLDER / f"pie_miner_{start_block}_{end_block}.pdf",
        bbox_inches="tight",
    )

    plt.show()
