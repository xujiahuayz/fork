import time
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import SUM_HASH_RATE, DATA_FOLDER
from fork_env.integration import fork_rate


def compute_rate(args):
    distribution, block_propagation_time, n = args

    rate = fork_rate(
        proptime=block_propagation_time,
        sum_lambda=SUM_HASH_RATE,
        n=n,
        dist=distribution,
        epsrel=1e-9,
        epsabs=1e-16,
        limit=130,
        limlst=10,
    )
    rate_dict = {
        "distribution": distribution,
        "block_propagation_time": block_propagation_time,
        "n": n,
        "rate": rate,
    }
    print(rate_dict)
    return rate_dict


if __name__ == "__main__":
    distributions = [
        "exp"
        # , "log_normal", "lomax"
    ]
    block_propagation_times = [0.87, 7.12, 8.7, 1_000]
    ns = range(2, 31)
    combinations = product(distributions, block_propagation_times, ns)

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    # save rates to json
    with open(DATA_FOLDER / "rates_integration_exp.json", "w") as f:
        json.dump(rates, f)
