from typing import Callable, Iterable

from joblib import Parallel, delayed
import numpy as np


def simulate_fork(
    n_miners: int,
    hash_distribution: Callable[[int], Iterable[float]],
) -> tuple[float, float]:
    hash_rates = hash_distribution(n_miners)
    sorted_mining_times = [
        np.random.exponential(1 / one_hash) for one_hash in hash_rates
    ]
    sorted_mining_times.sort()

    last_mining_time = sorted_mining_times[0]
    time_diff = sorted_mining_times[1] - last_mining_time
    return time_diff, last_mining_time


def simulate_fork_repeat(repeat: int, **kwargs) -> list[tuple[float, float | None]]:
    """
    Simulate a fork repeat times and return the results using Joblib for parallel processing.
    """
    results = Parallel(n_jobs=-1)(
        delayed(simulate_fork)(**kwargs) for _ in range(repeat)
    )
    return results  # type: ignore


if __name__ == "__main__":
    from fork_env.utils import expon_dist, lognorm_dist, lomax_dist, truncpl_dist

    x = simulate_fork(
        n_miners=10,
        hash_distribution=lambda n: truncpl_dist(hash_mean=1 / 60).rvs(size=n),
    )
    print(x)
