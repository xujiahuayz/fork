import pickle
import pandas as pd
from fork_env.constants import DATA_FOLDER, SUM_HASHES, FIGURES_FOLDER
from matplotlib import pyplot as plt

rates = pd.DataFrame(
    list(w[0]) + [w[1]]
    for w in pickle.load(open(DATA_FOLDER / "analytical_id.pkl", "rb"))
)

rates.columns = ["iid", "block_propagation_time", "sum_hash", "rate"]

# increase font size of plots
plt.rcParams.update({"font.size": 20})

for sum_hash in SUM_HASHES:
    df = rates[rates["sum_hash"] == sum_hash]
    df = df.sort_values(by="block_propagation_time")
    # plot iid true and iid false
    df_iid = df[df["iid"] == True]
    df_not_iid = df[df["iid"] == False]
    plt.plot(
        df_iid["block_propagation_time"],
        df_iid["rate"],
        label="i.i.d.",
        color="red",
        linestyle="--",
    )
    plt.plot(
        df_not_iid["block_propagation_time"],
        df_not_iid["rate"],
        label="independent",
        color="red",
    )
    plt.xlabel("block propagation time $\\Delta_0$ [s]")
    plt.ylabel("fork rate $C(\\Delta_0)$")

    # add legend on bottom right
    plt.legend(loc="lower right", title="assumption")

    plt.tight_layout()
    plt.savefig(FIGURES_FOLDER / f"fork_rate_id_{round(sum_hash,4)}.pdf")
    plt.show()
