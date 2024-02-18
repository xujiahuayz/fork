import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import DATA_FOLDER, SUM_HASH_RATE
from fork_env.integration import fork_rate


def compute_rate_log_lomax(args) -> tuple[tuple, float]:
    c, block_propagation_time, n = args

    the_key = (
        "lomax",
        block_propagation_time,
        n,
        SUM_HASH_RATE,
        c,
    )
    print(the_key)

    the_rate = fork_rate(
        proptime=block_propagation_time,
        sum_lambda=SUM_HASH_RATE,
        n=n,
        dist="lomax",
        c=c,
        epsrel=1e-9,
        epsabs=1e-16,
        limit=130,
    )
    print(the_key, the_rate)
    return the_key, the_rate


if __name__ == "__main__":
    # with open(DATA_FOLDER / "per_sigmal.pkl", "rb") as f:
    #     rates_c_dict = pickle.load(f)
    rates_c_dict = {}
    cs = [
        # 0.55,
        # 0.6,
        # 0.7,
        # 0.8,
        # 0.9,
        #
        1.01,
        1.02,
        1.05,
        1.15,
        1.2,
        1.4,
        1.6,
        1.8,
        2,
        2.2,
        2.5,
        3,
        3.5,
        4,
        5,
        6,
    ]

    block_propagation_times = [
        0.763,
        2.48,
        16.472,
        7.12,
        8.7,
        1_000,
    ]
    ns = range(2, 31)
    combinations = product(cs, block_propagation_times, ns)

    # # remove keys in rates_sigma_dict from combinations
    # combinations = [
    #     (c, block_propagation_time, n)
    #     for c, block_propagation_time, n in combinations
    #     if (
    #         "lomax",
    #         block_propagation_time,
    #         n,
    #         SUM_HASH_RATE,
    #         c,
    #     )
    #     not in rates_c_dict
    # ]
    if combinations:

        start_time = time.time()

        # Using ProcessPoolExecutor to parallelize computation
        with ProcessPoolExecutor() as executor:
            rates = list(executor.map(compute_rate_log_lomax, combinations))

        end_time = time.time()
        print(f"Computation completed in {end_time - start_time} seconds.")

        for rate in rates:
            if rate is not None:
                rates_c_dict[rate[0]] = rate[1]

        # save rates_sigmal_dict to pickle
        with open(DATA_FOLDER / "per_c.pkl", "wb") as f:
            pickle.dump(rates_c_dict, f)
