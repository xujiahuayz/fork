import numpy as np
import pickle

import pandas as pd

import matplotlib.pyplot as plt

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER, EMPIRICAL_PROP_DELAY

# from scripts.read_analytical_rates import rates

with open(DATA_FOLDER / "analytical_zerominers_mixed.pkl", "rb") as f:
    rates = pickle.load(f)

rates = pd.DataFrame(list(w[0]) + [w[1]] for w in rates)
rates.columns = ["distribution", "block_propagation_time", "n_zero", "rate"]

# increase font size of plots
plt.rcParams.update({"font.size": 18})
# make axes label farther from ticker label values
# export df to excel
for distribution in rates["distribution"].unique():
    df_distribution = rates[rates["distribution"] == distribution]

    for w in list(EMPIRICAL_PROP_DELAY.values()) + [100]:
        df_distribution_sorted = df_distribution.loc[
            df_distribution["block_propagation_time"] == w
        ]

        # do line plot rate against n_zero
        plt.plot(
            df_distribution_sorted["n_zero"],
            df_distribution_sorted["rate"],
            label=w,
        )

    plt.xlabel("number of zero miners $N_0$")
    plt.ylabel("fork rate $C(\Delta_0)$")
    # set y-axis range
    plt.ylim(0, 0.154)

    # set legend on the right outside of the plot with short handle length
    plt.legend(
        title="$\\Delta_0$",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        handlelength=0.5,
    )

    # Save the plot to a file
    plt.savefig(
        FIGURES_FOLDER / f"zerominer_mixed_{distribution}.pdf",
        bbox_inches="tight",
    )
    plt.show()
