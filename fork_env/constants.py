import csv

import pandas as pd

from fork_env.settings import PROJECT_ROOT

DATA_FOLDER = PROJECT_ROOT / "data"
FIGURES_FOLDER = PROJECT_ROOT / "figures"
TABLE_FOLDER = PROJECT_ROOT / "tables"

BITCOIN_MINER_PATH = DATA_FOLDER / "bitcoin_miner_bigquery.pkl.gz"
BITCOIN_MINER_JSON_PATH = DATA_FOLDER / "bitcoin_miner.json"
CLUSTER_PATH = DATA_FOLDER / "clusters.pkl.gz"
SIMULATED_FORK_RATES_PATH = DATA_FOLDER / "rates_simulated.jsonl.gz"
ANALYTICAL_FORK_RATES_PATH = DATA_FOLDER / "rates_analytical.pkl"
ANALYTICAL_FORK_RATES_PATH_STD = DATA_FOLDER / "rates_analytical_std.pkl"
SIMULATED_FORK_RATES_EMP_DIST = DATA_FOLDER / "rates_simulated_emp_dist.jsonl"
BLOCK_MINER_CLOVERPOOL_PATH = DATA_FOLDER / "block_miners_clover.pkl"

EMPRITICAL_FORK_RATE = 0.0041
BLOCK_WINDOW = 20_000
FIRST_START_BLOCK = 360_000  # 500_000


# DIST_KEYS = [
#     "exp",
#     #  "log_normal",
#     "lomax",
#     "trunc_power_law",
# ]

DIST_DICT = {
    "exp": {
        "label": "$\\text{Exp}(r)$",
        "color": "blue",
    },
    "log_normal": {
        "label": "$\\text{LN}(\\mu, \\sigma^2)$",
        "color": "orange",
    },
    # "lomax": {
    #     "label": "$\\text{Lomax}(c, \\ell)$",
    #     "color": "green",
    # },
    "trunc_power_law": {
        "label": "$\\text{TPL}(\\alpha, \\beta)$",
        "color": "red",
    },
    "empirical": {
        "label": "semi-empirical, i.i.d.",
        "color": "black",
    },
}

DIST_KEYS = list(DIST_DICT.keys())

"""# to obtain list of mining pool names 
merged_df = pd.read_pickle(DATA_FOLDER / "merged_df.pkl")
series = merged_df['miner_cluster'].value_counts()

lst = [idx for idx in series.index if isinstance(idx, str)]

# Open the CSV file in write mode
with open('Miner_country.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    for item in lst:
        writer.writerow([item])"""

# Define the country order to plot
COUNTRY_ORDER = [
    "China",
    "China other",
    "Japan",
    "India",
    "USA",
    "Canada",
    "Norway",
    "Czech Republic",
    "Germany",
    "Netherlands",
    "Poland",
    "Georgia",
    "Russia",
    "Sweden",
    "other",
]

# assign each country to a colour palette
COLOR_MAP = {
    "China": "Reds",
    "China other": "Reds",
    "USA": "Blues",
    "Russia": "Purples",
    "Netherlands": "Greens",
    "India": "gist_yarg",
    "Canada": "winter",
    "Czech Republic": "YlOrBr",
    "Georgia": "summer",
    "Norway": "bone",
    "Sweden": "PuRd",
    "Japan": "copper",
    "Poland": "pink",
    "Germany": "GnBu",
}

# hash_panel = pd.read_pickle(DATA_FOLDER / "hash_panel.pkl")
# # get the last row of hash panel
# hash_panel_last_row = hash_panel.iloc[-1]

# get the last number of total hash rate
SUM_HASH_RATE = 1 / 600
N_MINER = 35
HASH_STD = 0.0001

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
    0.5: 1,
    0.9: 3,
    0.99: 20,
}

BLOCK_PROP_TIMES = list(
    set(
        list(EMPIRICAL_PROP_DELAY.values())
        + [
            0.1,
            0.2,
            0.3,
            0.4,
            0.6,
            0.8,
            1,
            2,
            3,
            5,
            7,
            10,
            15,
            20,
            25,
            30,
            40,
            50,
            70,
            100,
            150,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ]
    )
)

BLOCK_PROP_TIMES.sort()


SUM_HASHES = [1e-3, 3e-3, SUM_HASH_RATE]
SUM_HASHES.sort()

VAR_HEADER_UNIT_MAP = {
    "start_block": {"header": r"start \#", "unit": "", "precision": None, "subtab": 1},
    "start_time": {"header": r"start time", "unit": "", "precision": None, "subtab": 1},
    "proptime_50": {"header": r"50\%", "unit": r"[s]", "precision": 2, "subtab": 1},
    "proptime_90": {"header": r"90\%", "unit": r"[s]", "precision": 2, "subtab": 1},
    "proptime_99": {"header": r"99\%", "unit": r"[s]", "precision": 2, "subtab": 1},
    "average_block_time": {
        "header": r"$\overline{\text{block time}}$",
        "unit": r"$t_{\text {min }}$ [s]",
        "precision": 1,
        "subtab": 1,
    },
    # "avg_logged_difficulty": {
    #     "header": r"$\overline{\ln(\text{difficulty})}$",
    #     "unit": "",
    #     "precision": 2,
    #     "subtab": 1,
    # },
    "fork_rate": {
        "header": r"fork rate",
        "unit": r"[\%]",
        "precision": 3,
        "subtab": 1,
    },
    "num_miners": {"header": r"miners", "unit": r"$N$", "precision": 0, "subtab": 1},
    "total_hash_rate": {
        "header": r"$\sum(\text{hash rate})$",
        "unit": r"$\Lambda$ [block/s]",
        "precision": 5,
        "subtab": 1,
    },
    "hash_mean": {
        "header": r"mean",
        "unit": r"$m$ [block/s]",
        "precision": 6,
        "subtab": 1,
    },
    "hash_std": {
        "header": r"std",
        "unit": r"$s$ [block/s]",
        "precision": 6,
        "subtab": 1,
    },
    "hash_skew": {
        "header": r"skewness",
        "unit": "",
        "precision": 2,
        "subtab": 1,
    },
    "hash_kurt": {
        "header": r"kurtosis",
        "unit": "",
        "precision": 2,
        "subtab": 1,
    },
    "hhi": {
        "header": r"hhi",
        "unit": "",
        "precision": 2,
        "subtab": 1,
    },
    "max_share": {
        "header": r"max share",
        "unit": r"[\%]",
        "precision": 2,
        "subtab": 1,
    },
    "exp_rate": {
        "header": r"$\text{Exp}(r)$",
        "unit": r"$r$",
        "precision": 0,
        "subtab": 2,
    },
    "log_normal_loc": {
        "header": r"\multicolumn{2}{c}{$\text{LN}(\mu, \sigma^2)$}",
        "unit": r"$\mu$",
        "precision": 2,
        "subtab": 2,
    },
    "log_normal_sigma": {
        "header": r"\multicolumn{2}{c}{$\text{LN}(\mu, \sigma^2)$}",
        "unit": r"$\sigma$",
        "precision": 2,
        "subtab": 2,
    },
    "truncpl_alpha": {
        "header": r"\multicolumn{2}{c}{$\text{TPL}(\alpha, \beta)$",
        "unit": r"$\alpha$",
        "precision": 2,
        "subtab": 2,
    },
    "truncpl_ell": {
        "header": r"\multicolumn{2}{c}{$\text{TPL}(\alpha, \beta)$",
        "unit": r"$\beta$",
        "precision": 0,
        "subtab": 2,
    },
    "difficulty": {
        "header": r"average difficulty",
        "unit": r"$[T$Hash/block]",
        "precision": 0,
        "subtab": 3,
    },
    "efficiency": {
        "header": r"average efficiency",
        "unit": r"[J/$T$Hash]",
        "precision": 2,
        "subtab": 3,
    },
    "total_hash_power": {
        "header": r"total hash power",
        "unit": r"[MW]",
        "precision": 0,
        "subtab": 3,
    },
    "waste_hash_ln": {
        "header": r"wasted hash",
        "unit": r"[block/s]",
        "precision": 5,
        "subtab": 3,
    },
    "wasted_power_ln": {
        "header": r"wasted power",
        "unit": r"[MW]",
        "precision": 0,
        "subtab": 3,
    },
    "waste_hash_tpl": {
        "header": r"wasted hash",
        "unit": r"[block/s]",
        "precision": 5,
        "subtab": 3,
    },
    "wasted_power_tpl": {
        "header": r"wasted power",
        "unit": r"[MW]",
        "precision": 0,
        "subtab": 3,
    },
}
