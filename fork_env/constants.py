import pandas as pd
from fork_env.settings import PROJECT_ROOT


DATA_FOLDER = PROJECT_ROOT / "data"
FIGURES_FOLDER = PROJECT_ROOT / "figures"
TABLE_FOLDER = PROJECT_ROOT / "tables"

BITCOIN_MINER_PATH = DATA_FOLDER / "bitcoin_miner.pkl"
BITCOIN_MINER_JSON_PATH = DATA_FOLDER / "bitcoin_miner.json"
CLUSTER_PATH = DATA_FOLDER / "clusters.pkl"
SIMULATED_FORK_RATES_PATH = DATA_FOLDER / "rates_simulated.jsonl"


DIST_KEYS = ["exp", "log_normal", "lomax"]

DIST_COLORS = {
    DIST_KEYS[0]: "blue",
    DIST_KEYS[1]: "orange",
    DIST_KEYS[2]: "green",
}

DIST_LABELS = {
    DIST_KEYS[0]: "$\\text{Exp}(r)$",
    DIST_KEYS[1]: "$\\text{LN}(\\mu, \\sigma^2)$",
    DIST_KEYS[2]: "$\\text{Lomax}(\\alpha, \\ell)$",
}

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate
SUM_HASH_RATE = hash_panel["total_hash_rate"].iloc[-1]
N_MINER = hash_panel["num_miners"].iloc[-1]
HASH_STD = hash_panel["hash_std"].iloc[-1]

# invstat.gpd
# Format:
# $1 + $2: unix timestamp (in milliseconds) of beginning and end of considered interval (usually 1h)
# $3: total number of INV entries received during period
# $5: 50% TX propagation delay (milliseconds)
# $7: 90% TX propagation delay (milliseconds)
# $9: 99% TX propagation delay (milliseconds)
# $11: 50% Block propagation delay (milliseconds)
# $13: 90% Block propagation delay (milliseconds)
# $15: 99% Block propagation delay (milliseconds)
# 1716033902225	1716037503164	102772821	0.5	7057	0.9	17683	0.99	28007	0.5	816	0.9	2666	0.99	14916	2024-05-18 12:05:02_225	2024-05-18 13:05:03_164

EMPIRICAL_PROP_DELAY = {
    0.5: 0.816,
    0.9: 2.666,
    0.99: 14.916,
}

NUMBER_MINERS_LIST = [2, 3, 4, 5, 8, 10, 20, 50, 100, 120, 150, 200]
