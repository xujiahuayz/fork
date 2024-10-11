import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from fork_env.constants import (
    DATA_FOLDER,
    EMPRITICAL_FORK_RATE,
    DIST_DICT,
    FIGURES_FOLDER,
)

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
block_dicts = hash_panel["block_dict"]
end_times = pd.to_datetime(hash_panel["end_time"])
emp_rate_in_percent = "{:.2%}".format(EMPRITICAL_FORK_RATE)


for proptime in ["50", "90", "99"]:
    # Create the plot
    fig, ax1 = plt.subplots()  # Use ax3 for the first y-axis
    # Create the second y-axis
    ax3 = ax1.twinx()

    ax1.plot(
        end_times,
        hash_panel["stale_rate"],
        linestyle=":",
        linewidth=3,
        alpha=0.8,
        label="actual fork rate",
    )

    ax3.set_title(f"to propagate to {proptime} percent of network")

    # Horizontal line for empirical fork rate
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

    # # write EMPRITICAL_FORK_RATE above the line
    # ax1.text(
    #     end_times[2],
    #     EMPRITICAL_FORK_RATE,
    #     emp_rate_in_percent,
    #     ha="right",
    #     va="bottom",
    # )

    for key, value in DIST_DICT.items():
        rates = [w[proptime][key] for w in block_dicts]
        ax1.plot(
            end_times,
            rates,
            label=value["label"],
            color=value["color"],
            linestyle="--" if key == "empirical" else "-",
        )
    ax1.set_ylabel("fork rate $C(\Delta_0)$")
    ax1.set_ylim(0.0001, 0.049)
    # ax1.set_yscale("log")

    # Format the x-axis as dates
    fig.autofmt_xdate()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=11))

    # Display the legend for the second y-axis
    if proptime == "50":
        ax1.legend(loc="upper left")
    # ax2.legend()

    # Save the plot
    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_time_series_{proptime}.pdf", bbox_inches="tight"
    )

    # Show the plot
    plt.show()
