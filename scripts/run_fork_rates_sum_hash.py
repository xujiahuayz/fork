import time
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from fork_env.constants import DATA_FOLDER
from fork_env.integration import fork_rate


def compute_rate_per_sumhash(args):
    distribution, sumhash, n = args

    rate = fork_rate(
        proptime=8.7,
        sum_lambda=sumhash,
        n=n,
        dist=distribution,
        epsrel=1e-9,
        epsabs=1e-16,
        limit=130,
        limlst=10,
    )
    rate_dict = {
        "distribution": distribution,
        "sumhash": sumhash,
        "n": n,
        "rate": rate,
    }
    print(rate_dict)
    return rate_dict


if __name__ == "__main__":
    distributions = ["exp", "log_normal", "lomax"]
    sumhashes = [
        1e-3,
        1e-2,
        1e-1,
        1e0,
        1e1,
    ]
    ns = range(2, 31)
    combinations = product(distributions, sumhashes, ns)

    start_time = time.time()
    rates = []

    # Using ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor() as executor:
        rates = list(executor.map(compute_rate_per_sumhash, combinations))

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time} seconds.")

    # save rates to json
    with open(DATA_FOLDER / "rates_integ_sumhash.json", "w") as f:
        json.dump(rates, f)
