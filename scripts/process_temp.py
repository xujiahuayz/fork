import json
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from fork_env.constants import DATA_FOLDER, FIGURES_FOLDER, SUM_HASH_RATE

# Load rates from json
with open(DATA_FOLDER / "rates_integ_log_normal.json", "r") as f:
    rates = json.load(f)


with open(DATA_FOLDER / "rates_integ_log_normal_add.json", "r") as f:
    rates_add = json.load(f)


with open("./scripts/tempa.txt", "r") as f:
    lines = f.readlines()
    # for line in lines:
    #     if "sigma" in line:
    #         # covert line to dict
    #         rate_dict = eval(line)

rates_new = [eval(line) for line in lines if "sigma" in line]

rates_add.extend(rates)
rates_add.extend(rates_new)

rates_sigmal_dict = {}

for w in rates_add:
    dict_key = (
        "log_normal",
        w["block_propagation_time"],
        w["n"],
        SUM_HASH_RATE,
        w["sigma"],
    )
    rates_sigmal_dict[dict_key] = w["rate"]

# save rates_sigmal_dict to pickle
import pickle


# open rates_sigmal_dict from pickle
with open("./scripts/tempa copy.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    if "log_normal" in line:
        # split string to get the key
        strings = line.split(") ")
        key = eval(strings[0] + ")")
        value = eval(strings[1])
        rates_sigmal_dict[key] = value

with open(DATA_FOLDER / "per_sigmal.pkl", "wb") as f:
    pickle.dump(rates_sigmal_dict, f)
