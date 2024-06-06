from matplotlib.colors import LogNorm
import numpy as np
import pickle

import pandas as pd

import matplotlib.pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER, SUM_HASHES
from scripts.read_analytical_rates import rates


DISTRIBUTIONS = ["exp", "log_normal", "lomax"]


labels = {
    "exp": "exponential",
    "log_normal": "log normal",
    "lomax": "lomax",
}
# increase font size of plots
plt.rcParams.update({"font.size": 18})
# make axes label farther from ticker label values
# rates = rates[rates["n"] < 500 & rates["n"] > 10]
rates = rates[(rates["n"] < 500) & (rates["n"] > 8)]
for sumhash in SUM_HASHES:
    df_hash = rates[rates["sum_hash"] == sumhash]
    # export df to excel
    for distribution in DISTRIBUTIONS:
        df_distribution = df_hash[df_hash["distribution"] == distribution]

        df_distribution_sorted = df_distribution.sort_values(
            by=["n", "block_propagation_time"]
        )

        # Create a pivot table to organize the data into a 2D grid, which is necessary for wireframe plots
        # keep nan values for the wireframe plot
        pivot_table = df_distribution.pivot_table(
            index="block_propagation_time",
            columns="n",
            values="rate",
            fill_value=np.nan,
        )

        # Extract the unique values of 'n' and 'block_propagation_time' to create meshgrid
        X_unique = list(pivot_table.columns)
        Y_unique = list(pivot_table.index)
        X, Y = np.meshgrid(
            X_unique, Y_unique
        )  # Use np.log to log the Y axis if necessary

        # The Z values are the fork rates. We need to align them with the X, Y grid
        Z = pivot_table.values

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        log_X = np.log2(X)
        log_Y = np.log10(Y)

        # add colorbar
        surf = ax.plot_surface(
            log_X,
            log_Y,
            Z,
            rstride=1,
            cstride=1,
            cmap="YlGn",
            edgecolor="none",
            alpha=0.8,
            vmin=0,
            vmax=1,
        )
        # add a think colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel("number of miners $N$", labelpad=10)
        ax.set_ylabel("block propagation delay \n $\\Delta_0$ [s]", labelpad=20)
        ax.set_zlabel("fork rate $C(\Delta_0)$", labelpad=10)

        ax.set_zlim(0, 1)

        # Adjust labels for logarithmic axis
        # ax.set_xlabel("number of miners $N$ (Log Scale)", labelpad=10)
        ax.set_xticks([4, 5, 6, 7])  # Set log-spaced ticks
        ax.set_yticks(np.linspace(-1, 3, 5))  # Set linear-spaced ticks

        ax.xaxis.set_major_formatter(lambda x, pos: f"$2^{{{int(x)}}}$")
        ax.yaxis.set_major_formatter(lambda x, pos: f"$10^{{{int(x)}}}$")

        title_value = round(sumhash, 5)
        plt.title(
            f"$\Sigma \lambda = {title_value}$ [$s^{{{-1}}}$]", loc="right", y=0.85
        )

        # Save the plot to a file
        plt.savefig(
            FIGURES_FOLDER / f"surface_{distribution}_{title_value}.pdf",
            bbox_inches="tight",
        )
        plt.show()
