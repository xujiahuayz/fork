import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from fork_env.constants import (
    EMPIRICAL_PROP_DELAY,
    SUM_HASHES,
    DATA_FOLDER,
    HASH_STD,
    N_MINER,
)
from fork_env.integration_ln import fork_rate_ln

# # open ANALYTICAL_FORK_RATES_PATH_STD and load the rates
# with open(DATA_FOLDER / "rates_analytical_std_ln.pkl", "rb") as f:
#     rates_sigma_dict = pickle.load(f)

# # change list to dict
# rates_sigma_dict = dict(rates_sigma_dict)


def compute_rate(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n, sumhash, std = args
    # print("processing", args)
    # if args are in the first element of the rates_sigma_dict, use the value
    try:
        # if (
        #     distribution,
        #     block_propagation_time,
        #     n,
        #     sumhash,
        #     std,
        # ) in rates_sigma_dict:
        #     the_rate_calculated = rates_sigma_dict[
        #         (distribution, block_propagation_time, n, sumhash, std)
        #     ]
        #     # only use this value if it is not None and not nan and > 1e-20

        #     the_rate = (
        #         the_rate_calculated
        #         if the_rate_calculated and the_rate_calculated > 1e-20
        #         else fork_rate_ln(
        #             proptime=block_propagation_time,
        #             sum_lambda=sumhash,
        #             n=n,
        #             std=std,
        #         )
        #     )
        #     print("using cached value", args)
        # else:
        the_rate = fork_rate_ln(
            proptime=block_propagation_time,
            sum_lambda=sumhash,
            n=n,
            std=std,
        )
    except Exception as e:
        print(f"Error in {args}: {e}")
        the_rate = np.nan
    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        ["log_normal"],
        list(EMPIRICAL_PROP_DELAY.values()),
        [
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            160,
            180,
            200,
            300,
            400,
            500,
        ]
        + [N_MINER],
        SUM_HASHES,
        list(np.logspace(-5, -2, num=31, base=10)) + [HASH_STD],
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(DATA_FOLDER / "rates_analytical_std_ln.pkl", "wb") as f:
        pickle.dump(rates, f)
