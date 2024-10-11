import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from fork_env.constants import (
    EMPIRICAL_PROP_DELAY,
    # SUM_HASHES,
    DATA_FOLDER,
    HASH_STD,
    N_MINER,
    SUM_HASH_RATE,
)
from fork_env.integration_ln import fork_rate_ln
from fork_env.integration_lomax import fork_rate_lomax

dist_dict = {
    "lomax": fork_rate_lomax,
    "log_normal": fork_rate_ln,
}


def compute_rate(args) -> tuple[tuple, float]:
    distribution, block_propagation_time, n, sumhash, std = args
    try:
        the_rate = dist_dict[distribution](
            proptime=block_propagation_time,
            sum_lambda=sumhash,
            n=n,
            std=std,
        )
    except Exception as e:
        # don't care -- too many prints otherwise
        # print(f"Error in {args}: {e}")
        the_rate = np.nan
    print(args, the_rate)
    return args, the_rate


if __name__ == "__main__":
    combinations = product(
        ["log_normal", "lomax"],
        set(list(EMPIRICAL_PROP_DELAY.values()) + [8.7]),
        set(
            [
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
            ]
            + [N_MINER]
        ),
        [SUM_HASH_RATE],
        # SUM_HASHES,
        set(list(np.logspace(-4.5, -2.3, num=30, base=10)) + [HASH_STD]),
    )

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    with open(DATA_FOLDER / "rates_analytical_heatmap.pkl", "wb") as f:
        pickle.dump(rates, f)
