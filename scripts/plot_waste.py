from fork_env.constants import SUM_HASH_RATE
from fork_env.integration_ln import waste_ln
from fork_env.integration_tpl import waste_tpl
from matplotlib import pyplot as plt

std_values = [
    0.00005,
    0.00006,
    0.00007,
    0.00008,
    0.00009,
    0.0001,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
    0.0006,
    0.0007,
    0.0008,
    0.0009,
    0.001,
]
# start a plot
plt.rcParams.update({"font.size": 15})
# fix y axis range
ylim_range = [0, 0.0016]
for key, waste_func in {
    "log_normal": waste_ln,
    "trunc_power_law": waste_tpl,
}.items():
    plt.figure()
    for num_miners in [10, 35, 60]:
        waste = [
            waste_func(
                n=num_miners,
                sum_lambda=SUM_HASH_RATE,
                std=std,
            )
            for std in std_values
        ]
        plt.plot(
            std_values,
            waste,
            label=f"{num_miners}",
            linestyle="-",
        )
    plt.ylim(ylim_range)

    plt.xlabel("standard deviation $s$ [s$^{-1}$]")
    plt.ylabel("wasted hash $\sum_{i \\neq k}\lambda$ [s$^{-1}$]")
    plt.legend(title="number of miners $N$")
    # tight layout
    plt.tight_layout()
    # save to file
    plt.savefig(f"figures/waste_{key}.pdf")
    plt.show()
