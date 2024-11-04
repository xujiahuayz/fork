import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from datetime import datetime
import re

from fork_env.constants import (
    DATA_FOLDER,
    BLOCK_WINDOW,
    FIRST_START_BLOCK,
    FIGURES_FOLDER,
    MINER_COUNTRY,
    COLOR_MAP,
)

# unpickle all dataframes
block_time_df = pd.read_pickle(DATA_FOLDER / "block_time_df.pkl")
agg_df = pd.read_pickle(DATA_FOLDER / "block_miners.pkl")
agg_df_2 = pd.read_pickle(DATA_FOLDER / "block_miners_2.pkl")

agg_df = pd.concat([agg_df, agg_df_2], ignore_index=True)

agg_df.set_index("height", inplace=True)

for idx in block_time_df.index:
    value_A = block_time_df.loc[idx, 'miner_cluster']
    if idx in agg_df.index: 
        value_B = agg_df.loc[idx, 'miner_cluster']

        # Check if value_A is in the dictionary and value_B is not
        if value_B in MINER_COUNTRY.keys() and value_A not in MINER_COUNTRY.keys():
            # Replace value in DataFrame B with value from DataFrame A
            block_time_df.loc[idx, 'miner_cluster'] = value_B

block_time_df['block_timestamp'] = block_time_df['block_timestamp'].dt.strftime('%Y-%m')

# Update BTC.com with its latest name "Cloverpool"
# Update SlushPool with its latest name "Braiins Pool"
block_time_df.loc[block_time_df["miner_cluster"] == "BTC.com", "miner_cluster" ] = "Cloverpool"
block_time_df.loc[block_time_df["miner_cluster"] == "SlushPool", "miner_cluster"] = "Braiins Pool"
block_time_df.loc[block_time_df["miner_cluster"] == "BTCC Pool", "miner_cluster"] = "BTCC"
block_time_df.loc[block_time_df["miner_cluster"] == "Huobi Pool", "miner_cluster"] = "Huobi"

# Label all unnamed miners as "other" for ease
block_time_df.loc[~block_time_df["miner_cluster"].isin(MINER_COUNTRY.keys()), "miner_cluster"] = "other"

value_counts = block_time_df["miner_cluster"].value_counts()

# function to extract shades from selected colour palettes
def sample_colormaps(cmap_names, num_intervals):
    return {name: [plt.get_cmap(name)(i) for i in np.linspace(0.30, 0.95, num_intervals)][::-1] for name in cmap_names}

# Get number of shades to extract for each country and sample those shades
country_counts = pd.Series(MINER_COUNTRY).value_counts()

sampled_colors = { 
            country: sample_colormaps([COLOR_MAP[country]], count)[COLOR_MAP[country]]
            for country, count in country_counts.items()
    }

# Assign shades to miners
country_miner_shade = {
    country: {
        miner: sampled_colors[country][i]
        for i, miner in enumerate(
            sorted([m for m, c in MINER_COUNTRY.items() if c == country])  # Alphabetical sorting of miner clusters
        )
    }
    for country in sampled_colors
}

# list of stacked bars to plot (periods on the x-axis)
period_grp = []
for start_block in range(
    FIRST_START_BLOCK,
    840_000,
    BLOCK_WINDOW,
):
    end_block = start_block + BLOCK_WINDOW - 1
    period_grp.append(f"{start_block}   \n({block_time_df.loc[start_block, 'block_timestamp']})")

# To track where each bar stack starts
block_bottom = np.zeros(len(period_grp))

period_data = []
China_other = ["Poolin", "BTC.TOP", "BW.COM", "Bixin", 
               "1THash", "1Hash", "BTC M6","BTC M4", "58COIN", 
               "Lubian", "BTC M7", "BTC M8", "BTC M1", "BTC M19", 
               "Huobi", "BTC M2", "BTC M3", "BTC M5"]

# traverse through each period from block_time_df with designated block window (20_000 in our case)
for start_block in range(
    FIRST_START_BLOCK,
    840_000,
    BLOCK_WINDOW,
):
    end_block = start_block + BLOCK_WINDOW - 1

    df_in_scope = block_time_df.loc[start_block:end_block]

    # Get the value counts for each period and map colors using the color_map
    value_counts = df_in_scope["miner_cluster"].value_counts()

    countries = [
        MINER_COUNTRY.get(miner, "other")
        for miner in value_counts.index
    ]

    # assign shades
    colors = [
        plt.get_cmap("Reds")(0.30) if miner in China_other else country_miner_shade.get(country, {}).get(miner, "gainsboro") 
        for miner, country in zip(value_counts.index, countries)
    ]
    
    # Add normalized value counts (to make the stacked bars add up to 100%)
    total_blocks = value_counts.sum() # - value_counts['other']
    normalized_counts = value_counts / total_blocks * 100
    
    # Create a DataFrame to track the period data
    period_df = pd.DataFrame({
        'normalized_count': normalized_counts,
        'color': colors,
        'country': countries,
    })

    period_data.append(period_df)

# Combine data from all periods into one DataFrame
combined_df = pd.concat(period_data, keys=period_grp, names=["Period"])

# Create a pivot table to stack the miners based on their country and cluster
pivot_df = combined_df.pivot_table(index="Period", columns=["country", "miner_cluster", "color"], values="normalized_count", fill_value=0, sort=True)

# Define the country order to plot 
country_order = ["China", "China other", "Japan", "India", "USA", "Canada", "Czech Republic", "Germany", "Netherlands", "Poland", "Russia", "Sweden", "other"]
miner_cluster_order = sorted(pivot_df.columns.get_level_values(1).astype(str))

# sort according to country first, followed by miner_cluster 
sorted_columns = sorted(
    pivot_df.columns,
    key=lambda x: (
        country_order.index(x[0]) if x[0] in country_order else -1,
        miner_cluster_order.index(x[1]) if x[1] in miner_cluster_order else -1
    )
)

# apply sorted order to pivot_df
pivot_df = pivot_df[sorted_columns]

# Plotting the stacked bar graph
fig, ax = plt.subplots(figsize=(16, 12))

other_label_added = False
china_other_added = False
# Iterate over each country and stack clusters of that country
for (country, miner_cluster, color), data in pivot_df.items():
    if data.sum() > 0:  # Only plot if there is data
       if data.sum() > 0:  # Only plot if there is data
        label = f"{miner_cluster} ({country})" if color != "gainsboro" else "Other (Unknown)"
        
        # Set hatch pattern for 'other' miner cluster instead of using the 'gainsboro' color
        if country == "other":
            # pass
            hatch = "."  # or "." for dot pattern
            ax.bar(
                pivot_df.index, data, color="white", edgecolor="black", hatch=hatch, linewidth=0.25,
                bottom=pivot_df.iloc[:, :pivot_df.columns.get_loc((country, miner_cluster, color))].sum(axis=1),
                label="Other (Unknown)" if not other_label_added else None
            )
            other_label_added = True  # Ensure only one legend entry for 'Other (Unknown)
        
        elif country == "China other":
            hatch = "."  # or "." for dot pattern
            ax.bar(
                pivot_df.index, data, color=color, hatch=hatch, 
                bottom=pivot_df.iloc[:, :pivot_df.columns.get_loc((country, miner_cluster, color))].sum(axis=1),
                label="Other (China)" if not china_other_added else None
            )
            china_other_added = True  # Ensure only one legend entry for 'Other (China)'

        else:
            ax.bar(
                pivot_df.index, data, color=color, 
                bottom=pivot_df.iloc[:, :pivot_df.columns.get_loc((country, miner_cluster, color))].sum(axis=1),
                label=label
            )

# rotate x-axis labels to be visible
ax.set_xticks(np.arange(len(period_grp)))  # Set ticks at each period position
ax.set_xticklabels(
    [label if i % 2 == 0 else "" for i, label in enumerate(period_grp)],
    rotation=45, ha='right', fontsize=13
)

# Customize the plot
ax.set_ylabel("percentage of blocks mined [%]", fontsize=18)
ax.set_xlabel("start block # (start time)", fontsize=18)
ax.legend(title="miner (country)", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)

# save to file
plt.savefig(
    FIGURES_FOLDER / "plot_miner_stacked.pdf", # "plot_miner_stacked_.pdf",
    bbox_inches="tight",
)
