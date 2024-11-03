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

small_x = 180
small_y = 0.38
# shade out a square of x=0, y=0 to x=100, y=0.14 with grey color
# plt.fill_between(
#     [0, small_x, small_x, 0],
#     [0, 0, small_y, small_y],
#     color="grey",
#     alpha=0.1,
# )


for i, sum_hash in enumerate(SUM_HASHES):
    df = rates[rates["sum_hash"] == sum_hash]
    df = df.sort_values(by="block_propagation_time")
    # plot iid true and iid false
    # df_iid = df[df["iid"] == True]
    # df_not_iid = df[df["iid"] == False]

    for j, id in enumerate([None, False, True]):
        # get the fork rate for id is None
        df_iid = df[(df["iid"].isnull()) if id is None else (df["iid"] == id)]

        # plt.plot(
        #     df_not_iid["block_propagation_time"],
        #     df_not_iid["rate"],
        #     label="independent",
        #     color="black",
        # )
        plt.plot(
            df_iid["block_propagation_time"],
            df_iid["rate"],
            label=["frequentist", "bayesian, independent", "bayesian, iid"][j],
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
        label="$p(0\\vert \\{\\lambda_i\\})$",
        # \\Lambda \\cdot \\left[1-  \\sum_i \\left(\\frac{b_i}{B} \\right)^2 \\right] $",
        # linewidth=2,
    )

    plt.text(
        850,
        df_iid["rate"].iloc[-1] + [-0.05, 0.08, 0.11][i],
        f"$\Lambda = {round(sum_hash,4)}$",
        horizontalalignment="left",
        verticalalignment="top",
    )
    plt.xlabel("block propagation time $\\Delta_0$ [s]")
    plt.ylabel("fork rate $C(\\Delta_0)$")
    # yaxis
    plt.xlim(0, 1100)
    plt.ylim(0, 1.19)

    # add legend only once, no border
    if i == 0:
        plt.legend(loc="upper left", frameon=False, handlelength=1)

    # add legend on bottom right

    # plt.show()


plt.tight_layout()
plt.savefig(FIGURES_FOLDER / f"fork_rate_id.pdf")

plt.show()


# sum_hash = SUM_HASHES[1]
# df = rates[rates["sum_hash"] == sum_hash]
# df = df.sort_values(by="block_propagation_time")
# df_not_iid = df[df["iid"].isnull()]


# plt.fill_between(
#     [0, small_x, small_x, 0],
#     [0, 0, small_y, small_y],
#     color="grey",
#     alpha=0.1,
# )

# plt.plot(
#     [0, small_x],
#     [
#         0,
#         sum_hash * (1 - sum((b / BLOCK_WINDOW) ** 2 for b in BIS)) * small_x,
#     ],
#     color="blue",
#     linewidth=2,
# )
# # write d c / d t = 0 in parrallel to the theoretical line
# plt.text(
#     0,
#     0.25,
#     "$\\left( \\frac{d C(\\Delta_0)}{d \\Delta_0} \\right)_{\\Delta_0 = 0} = \\Lambda \\cdot \\left[1-  \\sum_i \\left(\\frac{b_i}{B} \\right)^2 \\right] $",
#     horizontalalignment="left",
#     verticalalignment="top",
#     rotation=37,
#     color="blue",
# )
# plt.plot(
#     df_not_iid["block_propagation_time"],
#     df_not_iid["rate"],
#     # label="i.i.d.",
#     color="black",
# )

# plt.text(
#     230,
#     small_y - 0.12,
#     f"$\Lambda = {round(sum_hash,4)}$",
#     horizontalalignment="left",
#     verticalalignment="top",
# )

# # plot a line y = k * x, k = 1 - sum(bis^2)/sum(bis)^2


# plt.xlabel("block propagation time $\\Delta_0$ [s]")
# plt.ylabel("fork rate $C(\\Delta_0)$")
# # yaxis
# plt.xlim(0, small_x)
# plt.ylim(0, small_y)

# plt.tight_layout()
# plt.savefig(FIGURES_FOLDER / f"fork_slope.pdf")
