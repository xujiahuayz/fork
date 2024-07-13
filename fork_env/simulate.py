from typing import Callable, Iterable

from joblib import Parallel, delayed
import numpy as np


def simulate_fork(
    n_miners: int,
    hash_distribution: Callable[[int], Iterable[float]],
    block_propagation_time: float,
) -> tuple[bool, float]:
    hash_rates = hash_distribution(n_miners)
    sorted_mining_times = [np.random.exponential(1 / miner) for miner in hash_rates]
    sorted_mining_times.sort()

    last_mining_time = sorted_mining_times[0]
    time_diff = sorted_mining_times[1] - last_mining_time
    return time_diff < block_propagation_time, last_mining_time


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
    return np.mean([result[0] for result in results])
