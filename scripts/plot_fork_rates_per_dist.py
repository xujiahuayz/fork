import json
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER, SUM_HASH_RATE


with open(DATA_FOLDER / "rates_integ.json", "r") as f:
    rates = json.load(f)


df = pd.DataFrame(rates)
df = df[df["sumhash"] == SUM_HASH_RATE]

DISTRIBUTIONS = ["exp", "log_normal", "lomax"]


labels = {
    "exp": "exponential",
    "log_normal": "log normal",
    "lomax": "lomax",
}
# increase font size of plots
plt.rcParams.update({"font.size": 18})
# make axes label farther from ticker label values


# export df to excel
for distribution in DISTRIBUTIONS:
    df_distribution = df[df["distribution"] == distribution]

    df_distribution_sorted = df_distribution.sort_values(
        by=["n", "block_propagation_time"]
    )

    # Create a pivot table to organize the data into a 2D grid, which is necessary for wireframe plots
    pivot_table = df_distribution.pivot_table(
        index="block_propagation_time", columns="n", values="rate"
    )

    # Extract the unique values of 'n' and 'block_propagation_time' to create meshgrid
    X_unique = np.sort(df_distribution["n"].unique())
    Y_unique = np.sort(df_distribution["block_propagation_time"].unique())
    X, Y = np.meshgrid(X_unique, Y_unique)  # Use np.log to log the Y axis if necessary

    # The Z values are the fork rates. We need to align them with the X, Y grid
    Z = pivot_table.values

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    # add colorbar
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap="YlGn", edgecolor="none", alpha=0.8
    )
    # add a think colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel("number of miners $n$", labelpad=10)
    ax.set_ylabel("block propagation time $\\Delta_0$", labelpad=15)
    ax.set_zlabel("fork rate $C(\Delta_0)$", labelpad=10)

    ax.set_zlim(0, 1)

    # Save the plot to a file
    plt.savefig(FIGURES_FOLDER / f"surface_{distribution}.pdf", bbox_inches="tight")
    plt.show()