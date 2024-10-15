import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from fork_env.constants import DATA_FOLDER, DIST_DICT, FIGURES_FOLDER, BLOCK_WINDOW

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
block_dicts = hash_panel["block_dict"]
end_times = pd.to_datetime(hash_panel["end_time"])

# hash_panel["fork_rate_std"] = (
#     (hash_panel["fork_rate"] / 100 * (1 - hash_panel["fork_rate"] / 100))
#     / BLOCK_WINDOW
# ).sqrt()
# calculate the standard deviation of fork rate, (p*(1-p)/Block_window)^0.5
hash_panel["fork_rate_std"] = (
    (hash_panel["fork_rate"] / 100 * (1 - hash_panel["fork_rate"] / 100)) / BLOCK_WINDOW
).pow(0.5) * 100


# hash_panel["stale_rate"] / [w["50"]["proptime"] for w in block_dicts]
# [w["50"]["empirical"] for w in block_dicts] / hash_panel["stale_rate"]
plt.rcParams.update({"font.size": 15})

for proptime in ["50", "90", "99"]:
    # Create the plot
    fig, ax1 = plt.subplots()  # Use ax3 for the first y-axis
    # Create the second y-axis
    ax3 = ax1.twinx()

    ax1.plot(
        end_times,
        hash_panel["fork_rate"],
        linestyle=":",
        linewidth=3,
        alpha=0.8,
        label="actual fork rate",
        color="purple",
    )

    # plot 90% confidence interval

    ax1.fill_between(
        end_times,
        hash_panel["fork_rate"] - hash_panel["fork_rate_std"] * 1.645,
        hash_panel["fork_rate"] + hash_panel["fork_rate_std"] * 1.645,
        color="purple",
        alpha=0.1,
        # label="90% confidence interval",
    )

    ax3.set_ylabel("block propagation time $\Delta_0$ (s)", color="olive")
    ax3.plot(
        end_times,
        [w[proptime]["proptime"] for w in block_dicts],
        color="olive",
        linewidth=3,
        alpha=0.3,
    )
    ax3.tick_params(axis="y", colors="olive")  # Make y-axis ticks cyan

    # Set y-axis limits
    ax3.set_ylim(0.02, 28)

    # Make the left y-axis cyan
    ax3.spines["right"].set_color("olive")

    for key, value in DIST_DICT.items():
        rates = [w[proptime][key] * 100 for w in block_dicts]
        ax1.plot(
            end_times,
            rates,
            label=value["label"],
            color=value["color"],
            linestyle="--" if key == "empirical" else "-",
        )
    ax1.set_ylabel("fork rate $C(\Delta_0)$")
    ax1.set_ylim(0.0001, 5)

    # Format the x-axis as dates
    fig.autofmt_xdate()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=9))

    # Display the legend for the second y-axis
    if proptime == "50":
        ax1.legend(loc="upper left")

    # Save the plot
    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_time_series_{proptime}.pdf", bbox_inches="tight"
    )

    # Show the plot
    plt.show()
