# generate N hash rates based on a distribution
#  log-normal distribution

from typing import Callable

import numpy as np

from fork_env.constants import SUM_HASH_RATE
from fork_env.miners import Dlt, Miner


def simulate_fork(n: int, hash_distribution: Callable, block_propagation_time: float) -> tuple[bool, float | None]:

    hash_rates = hash_distribution(n)
    hash_rates_scaled = SUM_HASH_RATE * hash_rates / np.sum(hash_rates)


    miners = {i: Miner(id=i, hash_rate=hash_rates_scaled[i]) for i in range(n)}

    dlt = Dlt(miners=miners, block_propagation_time=block_propagation_time)
    fork_exists = dlt.fork_created()
    return fork_exists, dlt.last_mining_time
    

