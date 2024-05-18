import pandas as pd
from fork_env.settings import PROJECT_ROOT


DATA_FOLDER = PROJECT_ROOT / "data"
FIGURES_FOLDER = PROJECT_ROOT / "figures"
TABLE_FOLDER = PROJECT_ROOT / "tables"

BITCOIN_MINER_PATH = DATA_FOLDER / "bitcoin_miner.pkl"
BITCOIN_MINER_JSON_PATH = DATA_FOLDER / "bitcoin_miner.json"
CLUSTER_PATH = DATA_FOLDER / "clusters.pkl"

hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")

# get the last number of total hash rate
SUM_HASH_RATE = hash_panel["total_hash_rate"].iloc[-1]
N_MINER = hash_panel["num_miners"].iloc[-1]
HASH_STD = hash_panel["hash_std"].iloc[-1]
