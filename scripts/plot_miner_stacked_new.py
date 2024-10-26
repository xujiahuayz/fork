import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from fork_env.constants import (
    DATA_FOLDER,
    BLOCK_WINDOW,
    FIRST_START_BLOCK,
    FIGURES_FOLDER,
)

# unpickle block_time_df
block_time_df = pd.read_pickle(DATA_FOLDER / "block_time_df.pkl")

# country of origin of each selected mining pool.
country_map = {
    # https://slashdot.org/software/mining-pools/ 
    "Foundry USA": "USA", 
    "Antpool": "Mainland China",
    "F2Pool": "Mainland China",
    "ViaBTC": "Mainland China",
    "Binance Pool": "Mainland China", 
    "Mara Pool": "USA", # https://cointelegraph.com/news/marathon-digital-made-in-usa-bitcoin-blocks 
    "Braiins Pool": "British Virgin Islands", # https://braiins.com/legal/privacy-policy
    "Poolin": "Mainland China", 
    "BTC M4": "Mainland China",
    "SBI Crypto": "Japan",
    "BTC.com": "Mainland China",
    "BTCC Pool": "Hong Kong", # https://www.coindesk.com/markets/2018/11/06/crypto-exchange-btcc-is-closing-its-mining-pool-business/
    "1Hash": "Mainland China", # https://www.businessinsider.com/bitcoin-pools-miners-ranked-2016-6#12-1hash--079-7 
    "BTC.TOP": "Mainland China", 
    "BitFury": "Amsterdam", # https://bitfury.com/about 
    "KnCMiner": "Sweden", # https://cointelegraph.com/news/knc-miner-slush-pool-bitfury-at-odds-over-block-size-increase 
    "BitClub Network": "USA", # https://www.justice.gov/usao-nj/pr/nevada-man-admits-money-laundering-and-tax-offenses-related-bitclub-network-fraud-scheme 
    "1THash": "Mainland China", 
    "BTC M6": "Mainland China",
}

# assign each country to a colour palette
country_color_map = {
    "USA":"Blues", 
    "Mainland China": "Reds",
    "Hong Kong": "Purples",
    "British Virgin Islands": "YlOrBr",
    "Japan": "spring",
    "Amsterdam": "summer",
    "Sweden": "pink",
}

# define the country order to plot
country_order = ["Mainland China", "USA", "Hong Kong", "British Virgin Islands", "Amsterdam", "Japan", "Sweden", "other"]

# function to extract shades from selected colour palettes
def sample_colormaps(cmap_names, num_intervals):
    return {name: [plt.get_cmap(name)(i) for i in np.linspace(0.26, 1, num_intervals)][::-1] for name in cmap_names}

# Get number of shades to extract for each country and sample those shades
country_counts = pd.Series(country_map).value_counts()
sampled_colors = {
    country: sample_colormaps([country_color_map[country]], count)[country_color_map[country]]
    for country, count in country_counts.items()
}

# Assign shades to miners
country_miner_shade = {
    country: {
        miner: sampled_colors[country][i % len(sampled_colors[country])]
        for i, miner in enumerate(
            sorted([m for m, c in country_map.items() if c == country])  # Alphabetical sorting of miner clusters
        )
    }
    for country in sampled_colors
}
print(country_miner_shade)
        

# list of bars to plot (periods on x-axis)
period_grp = []
for start_block in range(
    FIRST_START_BLOCK,
    870_000,
    BLOCK_WINDOW,
):
    end_block = start_block + BLOCK_WINDOW - 1
    period_grp.append(f"{start_block}")

# To track where each bar stack starts
block_bottom = np.zeros(len(period_grp))

period_data = []

# traverse through each row from block_time_df
for start_block in range(
    FIRST_START_BLOCK,
    870_000,
    BLOCK_WINDOW,
):
    end_block = start_block + BLOCK_WINDOW - 1

    df_in_scope = block_time_df.loc[start_block:end_block]

    # Get the value counts for the pie plot and map colors using the color_map
    value_counts = df_in_scope["miner_cluster"].value_counts()

    countries = [
        country_map.get(miner, "other")
        for miner in value_counts.index
    ]
    
    # to accommodate for miner_clusters other than those listed above (see top)
    miner_grouping = [
        miner if miner in country_map.keys() else "Other"
        for miner in value_counts.index

    ]

    # assign shades
    colors = [
        country_miner_shade.get(country, {}).get(miner, "gainsboro") 
        for miner, country in zip(value_counts.index, countries)
    ]
    
    # Add normalized value counts (to make the stacked bars add up to 100%)
    total_blocks = value_counts.sum()
    normalized_counts = value_counts / total_blocks * 100
        
    # Create a DataFrame to track the period data
    period_df = pd.DataFrame({
        'normalized_count': normalized_counts,
        'color': colors,
        'country': countries,
        'grouping': miner_grouping,
        # 'code': country_code,
    })

    period_data.append(period_df)

# Combine data from all periods into one DataFrame
combined_df = pd.concat(period_data, keys=period_grp, names=["Period"])

# Create a pivot table to stack the miners based on their country and cluster
pivot_df = combined_df.pivot_table(index="Period", columns=["country", "miner_cluster", "color"], values="normalized_count", fill_value=0, sort=True)

# Manually sort the columns according to the custom country order
country_col = pivot_df.columns.get_level_values(0)

# Create a categorical variable for country with the defined order
country_cat = pd.Categorical(country_col, categories=country_order, ordered=True)

# Sort the DataFrame columns based on the custom order (if not the order would be alphabetically sorted by default)
sorted_columns = pivot_df.columns[np.argsort(country_cat)]
pivot_df = pivot_df[sorted_columns]

# Plotting the stacked bar graph
fig, ax = plt.subplots(figsize=(12, 8))

other_label_added = False
# Iterate over each country and stack clusters of that country
for (country, miner_cluster, color), data in pivot_df.items():
    if data.sum() > 0:  # Only plot if there is data
        label = f"{miner_cluster} ({country})" if color != "gainsboro" else "Other (Unknown)"
        if country == "other" and other_label_added:
            ax.bar(pivot_df.index, data, color=color, bottom=pivot_df.iloc[:, :pivot_df.columns.get_loc((country, miner_cluster, color))].sum(axis=1))
        else:
            if country == "other":
                other_label_added = True
            ax.bar(pivot_df.index, data, label=label, color=color, bottom=pivot_df.iloc[:, :pivot_df.columns.get_loc((country, miner_cluster, color))].sum(axis=1))

# rotate x-axis labels to be visible
ax.set_xticklabels(period_grp, rotation=45, ha='right', fontsize=10)  # Adjust rotation and alignment

# Customize the plot
ax.set_ylabel("Percentage of Blocks Mined")
ax.set_xlabel("Block Period")
ax.set_title("Miner Cluster Distribution per Period")
ax.legend(title="Miner Clusters and Countries", bbox_to_anchor=(1.05, 1), loc='upper left')

# save to file
plt.savefig(
    FIGURES_FOLDER / "Miner Clusters and Countries.pdf",
    bbox_inches="tight",
)

plt.show()