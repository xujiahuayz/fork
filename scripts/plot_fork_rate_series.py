import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from fork_env.constants import DATA_FOLDER, DIST_DICT, FIGURES_FOLDER, BLOCK_WINDOW

def final_touch(title:str, ax1:plt.Axes) -> None:
    # Format the x-axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=x_label_bin))
    # Make x-axis labels vertical and centered on ticks
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, ha='center')

    plt.savefig(FIGURES_FOLDER / f"{title}.pdf", bbox_inches="tight")
    plt.show()

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
block_dicts = hash_panel["block_dict"]
end_times = pd.to_datetime(hash_panel["end_time"])

# fill out na
block_time_50 = pd.Series(w["50"]["proptime"] for w in block_dicts).fillna(method='ffill').fillna(method='bfill').tolist()
block_time_90 = pd.Series(w["90"]["proptime"] for w in block_dicts).fillna(method='ffill').fillna(method='bfill').tolist()
block_time_99 = pd.Series(w["99"]["proptime"] for w in block_dicts).fillna(method='ffill').fillna(method='bfill').tolist()

max_time = 33

ROLLING_WINDOW = 3

fork_rate_ma = hash_panel["fork_rate"].rolling(window=ROLLING_WINDOW, center=True).mean()

plt.rcParams.update({"font.size": 15})

small_y = 1.39
x_label_bin = 20
handlelen = 0.6
# Create the plot
fig, ax1 = plt.subplots()  # Use ax3 for the first y-axis

# fill a horizontal band between 0 and small_y across the x-axis
# small_x =
plt.axhspan(0, small_y, color="grey", alpha=0.1)

# Create the second y-axis
ax3 = ax1.twinx()
ax3.set_ylabel("block propagation time $\Delta_0$ [s]", color="olive")
ax3.set_ylim(0, max_time)

for i, proptime in enumerate(["50", "90", "99"]):

    ax3.plot(
        end_times,
        [w[proptime]["proptime"] for w in block_dicts],
        color="olive",
        linewidth=4,
        alpha=0.4,
    )
    ax3.tick_params(axis="y", colors="olive")  # Make y-axis ticks cyan
    # Make the left y-axis cyan
    ax3.spines["right"].set_color("olive")

    plt.text(
        0.02,
        [0.08, 0.35, 0.65][i],
        f"{proptime}%",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax1.transAxes,
        rotation=0,
        color="olive",
    )

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

    if i == 0:
        # move upper center even a bit further up
        ax1.legend(loc="upper center", frameon=False, handlelength=0.7, ncol=3, bbox_to_anchor=(0.56, 1.05))

ax1.set_ylabel("fork rate $C(\Delta_0)$ [%]")
ax1.set_ylim(0, 5.2)

final_touch("fork_rate_time_series_full", ax1)



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
    linewidth=4,
    alpha=0.15,
    label="historically measured fork rate",
    color="purple",
)

# plot the same as above again but with centered moving average of hash_panel["fork_rate"]
ax1.plot(
    end_times,
    fork_rate_ma,
    color="purple",
    linestyle=":",
    # alpha=0.7,
    # label="historically measured fork rate \n(5-period centered moving average)",
    linewidth=1,
)

ax3.set_ylabel("block propagation time $\Delta_0$ [s]", color="olive")

# for i, proptime in enumerate(["50", "90", "99"]):

ax3.plot(
    end_times,
    block_time_50,
    color="olive",
    linewidth=4,
    alpha=0.4,
    label="$\Delta_0$ to propagate \n50% of network",
)
ax3.tick_params(axis="y", colors="olive")  # Make y-axis ticks cyan


# Set y-axis limits
ax3.set_ylim(0, small_y / 5.2 * max_time)
ax1.set_ylim(0, small_y)

ax1.legend(loc="upper right", frameon=False, handlelength=handlelen)
ax3.legend(loc="right", frameon=False, handlelength=handlelen)

final_touch("fork_rate_time_series_small", ax1)



# create a plot with two y-axes
fig, ax1 = plt.subplots()

ax1.plot(
    end_times,
    1 - hash_panel["fork_rate"] / 100 / (hash_panel["total_hash_rate"] * block_time_50),
    color="purple",
    label=f"implied with \nhistorically measured fork rate",
    linewidth=4,
    alpha=0.15,
)

ax1.plot(
    end_times,
    1 - hash_panel["fork_rate"].rolling(window=ROLLING_WINDOW, center=True).mean() / 100 / (hash_panel["total_hash_rate"].rolling(window=ROLLING_WINDOW, center=True).mean() * pd.Series(block_time_50).rolling(window=ROLLING_WINDOW, center=True).mean()),
    linewidth=1,
    color="purple",
    linestyle=":",
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
# ax1.axhline(y=1, color="black", linestyle="--", linewidth=0.5)
ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.5)


ax1.set_ylim(-0.09, 1.09)
ax1.set_ylabel("$\it HHI$")
ax1.legend(loc="upper left", frameon=False, handlelength=handlelen)

final_touch("fork_rate_time_series_hhi", ax1)




fig, ax1 = plt.subplots()

# fill bands for block propagation time
# different shades of grey for 0-50%, 50-90%, 90-99%, 99-100%

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
    linewidth=4,
    alpha=0.15,
)

ax1.plot(
    end_times,
    (hash_panel["fork_rate"]
    / 100
    / (1 - hash_panel["hhi"])
    / hash_panel["total_hash_rate"]).rolling(window=ROLLING_WINDOW, center=True).mean(),
    color="purple",
    linestyle=":",
)

for i, proptime in enumerate(["50", "90", "99"]):
    plt.text(
        0.01,
        [0.25, 0.55, 0.85][i],
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
final_touch("fork_rate_time_series_delta0", ax1)
