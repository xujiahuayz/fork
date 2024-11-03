import pickle
import pandas as pd
from fork_env.constants import DATA_FOLDER, SUM_HASHES, FIGURES_FOLDER, BLOCK_WINDOW
from matplotlib import pyplot as plt

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
BIS = hash_panel["bis"].iloc[-1]


rates = pd.DataFrame(
    list(w[0]) + [w[1]]
    for w in pickle.load(open(DATA_FOLDER / "analytical_id.pkl", "rb"))
)

rates.columns = ["iid", "block_propagation_time", "sum_hash", "rate"]
# increase font size of plots
plt.rcParams.update({"font.size": 15})

small_x = 190


for i, sum_hash in enumerate(SUM_HASHES):
    df = rates[rates["sum_hash"] == sum_hash]
    df = df.sort_values(by="block_propagation_time")

    for j, id in enumerate([None, False, True]):
        # get the fork rate for id is None
        df_iid = df[(df["iid"].isnull()) if id is None else (df["iid"] == id)]

        plt.plot(
            df_iid["block_propagation_time"],
            df_iid["rate"],
            label=["frequentist", "Bayesian, INID", "Bayesian, i.i.d."][j],
            color="black",
            alpha=0.4,
            linewidth=2,
            linestyle=["-", ":", "--"][j],
        )

    plt.plot(
        [0, small_x],
        [
            0,
            sum_hash * (1 - sum((b / BLOCK_WINDOW) ** 2 for b in BIS)) * small_x,
        ],
        color="red",
        label="$\Delta_0 \cdot p(0\\vert \\{\\lambda_i\\})$",
    )

    plt.text(
        730,
        df_iid["rate"].iloc[-1] + [-0.09, 0.08, 0.11][i],
        f"$\Lambda = {round(sum_hash,4)}$ [s$^{{-1}}$]",
        horizontalalignment="left",
        verticalalignment="top",
    )
    plt.xlabel("block propagation time $\\Delta_0$ [s]")
    plt.ylabel("fork rate $C(\\Delta_0)$")
    plt.xlim(0, 1100)
    plt.ylim(0, 1.19)

    # add legend only once, no border
    if i == 0:
        plt.legend(loc="upper left", frameon=False, handlelength=1)


plt.tight_layout()
plt.savefig(FIGURES_FOLDER / f"fork_rate_id.pdf")

plt.show()
