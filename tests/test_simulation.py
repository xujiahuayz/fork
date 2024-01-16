# simulate 9999 times and save results

import numpy as np
from fork_env.constants import SUM_HASH_RATE
from fork_env.simulate import simulate_fork_repeat


def test_simulate_fork():
    """
    test that average last mining time of 1000 simulations is close to 1 / SUM_HASH_RATE with pytest
    """
    results = simulate_fork_repeat(
        repeat=99_999,
        n=2,
        hash_distribution=lambda n: np.random.lognormal(mean=0, sigma=1, size=n),
        block_propagation_time=0.1,
    )

    # calculate the mean of the last mining time
    last_mining_times = np.mean(
        [result[1] for result in results if result[1] is not None]
    )
    assert abs(last_mining_times * SUM_HASH_RATE - 1) < 0.1  # type: ignore
