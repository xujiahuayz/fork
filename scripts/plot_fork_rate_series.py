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

for proptime in ["50", "90", "99"]:
    # Create the plot
    plt.figure()
    plt.title(f"to propogate to {proptime} percent of network")
    plt.ylabel("fork rate $C(\Delta_0)$")

    # Horizontal line for empirical fork rate
    plt.axhline(y=EMPRITICAL_FORK_RATE, color="gray", linestyle=":")

    for key, value in DIST_DICT.items():
        rates = [w[proptime][key] for w in block_dicts]
        plt.plot(
            end_times,
            rates,
            label=value["label"],
            color=value["color"],
            linestyle="--" if key == "empirical" else "-",
        )

    # Ensure x-axis is formatted as dates without time
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # Show only date

    # Set fewer ticks on x-axis
    plt.gca().xaxis.set_major_locator(
        MaxNLocator(nbins=6)
    )  # Adjust number of ticks as needed

    # Set y-axis limits
    plt.ylim(0, 0.05)

    # Display the legend
    plt.legend()

    # Save the plot
    plt.savefig(
        FIGURES_FOLDER / f"fork_rate_time_series_{proptime}.pdf", bbox_inches="tight"
    )

    # Show the plot
    plt.show()
