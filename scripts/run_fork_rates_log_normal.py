import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import DATA_FOLDER, SUM_HASH_RATE
from fork_env.integration import fork_rate


def compute_rate_log_normal(args):
    sigma, block_propagation_time, n = args

    rate = fork_rate(
        proptime=block_propagation_time,
        sum_lambda=SUM_HASH_RATE,
        n=n,
        dist="log_normal",
        sigma=sigma,
        epsrel=1e-9,
        epsabs=1e-16,
        limit=130,
        limlst=10,
    )
    rate_dict = {
        "sigma": sigma,
        "block_propagation_time": block_propagation_time,
        "n": n,
        "rate": rate,
    }
    print(rate_dict)
    return rate_dict


if __name__ == "__main__":
    sigmas = [
        0.3,
        0.5,
        1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.5,
        3,
        3.5,
        4,
        4.5,
        5,
        6,
        7,
        8,
        9,
        10,
    ]

    block_propagation_times = [
        0.86,
        0.87,
        7.12,
        8.7,
        1_000,
    ]
    ns = range(2, 31)
    combinations = product(sigmas, block_propagation_times, ns)

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate_log_normal, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    # save rates to json
    with open(DATA_FOLDER / "rates_integ_log_normal.json", "w") as f:
        json.dump(rates, f)
