import json
import time

from fork_env.constants import (
    # DIST_KEYS,
    SIMULATED_FORK_RATES_PATH,
    SUM_HASH_RATE,
    N_MINER,
)
from fork_env.simulate import simulate_fork_repeat
from fork_env.utils import expon_dist, lognorm_dist, lomax_dist, truncpl_dist
import gzip

HASH_MEAN = SUM_HASH_RATE / N_MINER

start_time = time.time()
# open file for writing by appending
with gzip.open(SIMULATED_FORK_RATES_PATH, "wt") as f:
    for n in [
        2,
        3,
        4,
        5,
        8,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        100,
        150,
        200,
        # 500,
        # 1000,
        # 1500,
        # 2000,
        # 5000,
        # 10000,
    ]:

        for dist, dist_func in {
            "truncate_power_law": truncpl_dist,
            "log_normal": lognorm_dist,
            # "lomax": lomax_dist,
            "exp": expon_dist,
        }.items():
            print(f"n: {n}, dist: {dist}")

            try:
                time_diffs = [
                    result[0]
                    for result in simulate_fork_repeat(
                        repeat=int(2e6),
                        n_miners=n,
                        hash_distribution=lambda n: dist_func(hash_mean=HASH_MEAN).rvs(
                            size=n
                        ),
                    )
                ]
            except ValueError as e:
                print(f"Error: {e}")
                # next iteration
                continue

            this_rate = {
                "distribution": dist,
                "n": n,
                "time_diffs": time_diffs,
            }
            # write to jsonl
            f.write(json.dumps(this_rate) + "\n")


time_taken = time.time() - start_time
print(f"Time taken: {time_taken}")
