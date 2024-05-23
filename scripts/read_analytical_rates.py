import pickle

import pandas as pd

from fork_env.constants import ANALYTICAL_FORK_RATES_PATH


rates = pd.DataFrame(
    list(w[0]) + [w[1]] for w in pickle.load(open(ANALYTICAL_FORK_RATES_PATH, "rb"))
)
rates.columns = ["distribution", "block_propagation_time", "n", "sum_hash", "rate"]
