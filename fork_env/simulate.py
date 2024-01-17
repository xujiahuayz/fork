from joblib import Parallel, delayed
from typing import Callable

import numpy as np

from fork_env.constants import SUM_HASH_RATE
from fork_env.dltenv import Dlt, Miner


def simulate_fork(
    n: int, hash_distribution: Callable, block_propagation_time: float
) -> tuple[bool, float | None]:
    hash_rates = hash_distribution(n)
    hash_rates = SUM_HASH_RATE * hash_rates / np.sum(hash_rates)

    miners = {i: Miner(id=i, hash_rate=hash_rates[i]) for i in range(n)}

    dlt = Dlt(miners=miners, block_propagation_time=block_propagation_time)
    fork_exists = dlt.fork_created()
    return fork_exists, dlt.last_mining_time


def simulate_fork_repeat(repeat: int, **kwargs) -> list[tuple[bool, float | None]]:
    """
    Simulate a fork repeat times and return the results using Joblib for parallel processing.
    """
    results = Parallel(n_jobs=-1)(
        delayed(simulate_fork)(**kwargs) for _ in range(repeat)
    )
    return results  # type: ignore


def get_fork_rate(repeat: int, **kwargs) -> float:
    """
    get fork rate from simulated results
    """
    results = simulate_fork_repeat(
        repeat=repeat,
        **kwargs,
    )
    return sum([result[0] for result in results]) / repeat
