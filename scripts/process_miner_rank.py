# open DATA_FOLDER / "miner_rank.json"

import json
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np


from fork_env.constants import DATA_FOLDER

with open(DATA_FOLDER / "miner_rank.json", "r") as f:
    miner_rank = json.load(f)

bitcoin_miners = miner_rank["data"]["items"][0]["minerModelList"]
miner_hashrate = [w["hashrateNumber"] for w in bitcoin_miners if w["coinType"] == "BTC"]

# sort the list miner_hashrate
miner_hashrate.sort(reverse=True)

miner_hashrate_df = pd.DataFrame(miner_hashrate, columns=["hashrate"])

miner_hashrate_df["hashrate"].plot(
    kind="hist",
    bins=100,
    density=True,
    cumulative=True,
    alpha=0.7,
    # color="blue",
    legend=True,
)

# Create plot with histogram
fig, ax1 = plt.subplots()
ax1.hist(
    miner_hashrate_df["hashrate"],
    bins=40,
    alpha=0.5,
    density=True,
    label="Histogram",
    cumulative=False,
)
ax1.set_xlabel("Hashrate")
ax1.set_ylabel("Density")
# ax1.set_xlim(left=0)

# # Create a secondary axis for the KDE plot
# ax2 = ax1.twinx()

# # Calculate the KDE with manual bandwidth selection
# bandwidth = np.std(miner_hashrate) * (len(miner_hashrate) ** (-1/5))  # Scott's Rule
# kde = gaussian_kde(miner_hashrate, bw_method=bandwidth)

# # Generate x values from 0 to maximum hashrate for plotting
# x = np.linspace(0, max(miner_hashrate), 1000)

# # Plot the KDE on the secondary Y-axis
# ax2.plot(x, kde(x), color='red', label='KDE')
# ax2.set_ylabel('KDE Density')

# # Setting the labels for legend
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Show the plot
plt.show()

# fit a power law to miner_hashrate
import powerlaw

fit = powerlaw.Fit(miner_hashrate)
print(fit.power_law.alpha, fit.power_law.xmin)

# plot the power law fit
fig2 = fit.plot_pdf(color="b", linewidth=2)
fit.power_law.plot_pdf(color="b", linestyle="--", ax=fig2)
plt.show()
