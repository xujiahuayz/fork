import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from fork_env.constants import DATA_FOLDER, DIST_DICT, FIGURES_FOLDER, BLOCK_WINDOW

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
block_dicts = hash_panel["block_dict"]
end_times = pd.to_datetime(hash_panel["end_time"])

max_time = 33
# calculate the standard deviation of fork rate, (p*(1-p)/Block_window)^0.5
hash_panel["fork_rate_std"] = (
    (hash_panel["fork_rate"] / 100 * (1 - hash_panel["fork_rate"] / 100)) / BLOCK_WINDOW
).pow(0.5) * 100


plt.rcParams.update({"font.size": 15})

small_y = 1.18
x_label_bin = 13
handlelen = 0.8
# Create the plot
fig, ax1 = plt.subplots()  # Use ax3 for the first y-axis

# fill a horizontal band between 0 and small_y across the x-axis
# small_x =
plt.axhspan(0, small_y, color="grey", alpha=0.1)


# Create the second y-axis
ax3 = ax1.twinx()

ax3.set_ylabel("block propagation time $\Delta_0$ (s)", color="olive")


for i, proptime in enumerate(["50", "90", "99"]):

    ax3.plot(
        end_times,
        [w[proptime]["proptime"] for w in block_dicts],
        color="olive",
        linewidth=5,
        alpha=0.4,
    )
    ax3.tick_params(axis="y", colors="olive")  # Make y-axis ticks cyan
    # Make the left y-axis cyan
    ax3.spines["right"].set_color("olive")

    plt.text(
        0.02,
        [0.13, 0.43, 0.72][i],
        f"{proptime}%",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax1.transAxes,
        rotation=0,
        color="olive",
    )

    # Set y-axis limits
    ax3.set_ylim(0, max_time)

    for key, value in DIST_DICT.items():
        if key != "empirical":
            rates = [w[proptime][key] * 100 for w in block_dicts]
            ax1.plot(
                end_times,
                rates,
                label=value["label"],
                color=value["color"],
                linestyle="--" if key == "empirical" else "-",
            )
    ax1.set_ylabel("fork rate $C(\Delta_0)$ [%]")
    ax1.set_ylim(0, 5)

    # Format the x-axis as dates
    fig.autofmt_xdate()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=x_label_bin))
    if i == 0:
        # two columns, no border
        ax1.legend(loc="upper center", frameon=False, handlelength=0.8)


plt.savefig(FIGURES_FOLDER / f"fork_rate_time_series_full.pdf", bbox_inches="tight")
plt.show()


# Create the plot
fig, ax1 = plt.subplots()  # Use ax3 for the first y-axis

plt.axhspan(0, small_y, color="grey", alpha=0.1)

# Create the second y-axis
ax3 = ax1.twinx()

tpl_label = "C(\\Delta_0)_{\\{\\lambda_i \\} \sim \\text{TPL}(\\alpha, \\beta)}"
freq_label = "C(\\Delta_0 \\vert \\{\\lambda_i = \\frac{b_i \cdot \Lambda}{B} \\})"
key = "trunc_power_law"
value = DIST_DICT[key]
rates = [w["50"][key] * 100 for w in block_dicts]
ax1.plot(
    end_times,
    rates,
    label=f"${tpl_label}$",
    color=value["color"],
    linestyle="--" if key == "empirical" else "-",
)
emp_rate = [w["50"]["empirical"] * 100 for w in block_dicts]
ax1.plot(
    end_times,
    emp_rate,
    label=f"${freq_label}$ ",
    color="black",
    alpha=0.8,
    # linestyle="--" if key == "empirical" else "-",
)
ax1.set_ylabel("fork rate $C(\Delta_0)$ [%]")

ax1.plot(
    end_times,
    hash_panel["fork_rate"],
    linestyle=":",
    linewidth=3,
    alpha=0.8,
    label="historically measured fork rate",
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

# for i, proptime in enumerate(["50", "90", "99"]):
block_time_50 = [w["50"]["proptime"] for w in block_dicts]
ax3.plot(
    end_times,
    block_time_50,
    color="olive",
    linewidth=5,
    alpha=0.4,
    label="$\Delta_0$ to propagate \n50% of network",
)
ax3.tick_params(axis="y", colors="olive")  # Make y-axis ticks cyan


# Set y-axis limits
ax3.set_ylim(0, small_y / 5 * max_time)


ax1.set_ylim(0, small_y)

# Format the x-axis as dates
fig.autofmt_xdate()
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax3.xaxis.set_major_locator(MaxNLocator(nbins=x_label_bin))

ax1.legend(loc="upper right", frameon=False, handlelength=handlelen)
ax3.legend(loc="right", frameon=False, handlelength=handlelen)
# # Display the legend for the second y-axis

# Save the plot
plt.savefig(FIGURES_FOLDER / f"fork_rate_time_series_small.pdf", bbox_inches="tight")

# Show the plot
plt.show()


# create a plot with two y-axes
fig, ax1 = plt.subplots()

ax1.plot(
    end_times,
    1 - hash_panel["fork_rate"] / 100 / (hash_panel["total_hash_rate"] * block_time_50),
    color="purple",
    label=f"implied with \nhistorically measured fork rate",
    linewidth=3,
    linestyle=":",
)

ax1.fill_between(
    end_times,
    1
    - (hash_panel["fork_rate"] - hash_panel["fork_rate_std"] * 1.645).clip(0.0001)
    / 100
    / (hash_panel["total_hash_rate"] * block_time_50),
    1
    - (hash_panel["fork_rate"] + hash_panel["fork_rate_std"] * 1.645)
    / 100
    / (hash_panel["total_hash_rate"] * block_time_50),
    color="purple",
    alpha=0.1,
    # label="90% confidence interval",
)


ax1.plot(
    end_times,
    hash_panel["hhi"],
    color="red",
    label="empirical with ${\lambda_i = \\frac{b_i \cdot \Lambda}{B}}$",
    linewidth=3,
    # linestyle=":",
)


# horizontal line at 1
ax1.axhline(y=1, color="black", linestyle="--", linewidth=0.5)

ax1.set_ylim(0, 1)
ax1.set_ylabel("$\it HHI$")

ax1.legend(loc="upper left", frameon=False, handlelength=handlelen)

fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(MaxNLocator(nbins=x_label_bin))

plt.savefig(FIGURES_FOLDER / f"fork_rate_time_series_hhi.pdf", bbox_inches="tight")


fig, ax1 = plt.subplots()

# fill bands for block propagation time
# different shades of grey for 0-50%, 50-90%, 90-99%, 99-100%
block_time_90 = [w["90"]["proptime"] for w in block_dicts]
block_time_99 = [w["99"]["proptime"] for w in block_dicts]
ax1.fill_between(end_times, 0, block_time_50, color="grey", alpha=0.9)
ax1.fill_between(
    end_times,
    block_time_50,
    block_time_90,
    color="grey",
    alpha=0.4,
)
ax1.fill_between(
    end_times,
    block_time_90,
    block_time_99,
    color="grey",
    alpha=0.3,
)
ax1.fill_between(
    end_times,
    block_time_99,
    33,
    color="grey",
    alpha=0.2,
)


ax1.plot(
    end_times,
    hash_panel["fork_rate"]
    / 100
    / (1 - hash_panel["hhi"])
    / hash_panel["total_hash_rate"],
    color="purple",
    label=f"implied with \nhistorically measured fork rate",
    linewidth=3,
    linestyle=":",
)

ax1.fill_between(
    end_times,
    (hash_panel["fork_rate"] - hash_panel["fork_rate_std"] * 1.645).clip(0.0001)
    / 100
    / (1 - hash_panel["hhi"])
    / hash_panel["total_hash_rate"],
    (hash_panel["fork_rate"] + hash_panel["fork_rate_std"] * 1.645)
    / 100
    / (1 - hash_panel["hhi"])
    / hash_panel["total_hash_rate"],
    color="purple",
    alpha=0.1,
    # label="90% confidence interval",
)

for i, proptime in enumerate(["50", "90", "99"]):
    plt.text(
        0.01,
        [0.2, 0.5, 0.8][i],
        f"{proptime}%",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax1.transAxes,
        rotation=0,
        color="black",
    )

ax1.set_ylim(0, max_time)
ax1.set_ylabel("block propagation time $\Delta_0$ [s]")

ax1.legend(loc="upper right", frameon=False, handlelength=handlelen)

fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(MaxNLocator(nbins=x_label_bin))

plt.savefig(FIGURES_FOLDER / f"fork_rate_time_series_delta0.pdf", bbox_inches="tight")
