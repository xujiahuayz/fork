import pickle

from fork_env.constants import (
    DATA_FOLDER,
    SUM_HASH_RATE,
    ANALYTICAL_FORK_RATES_PATH_STD,
)


# with open(DATA_FOLDER / "per_c_old.pkl", "rb") as f:
#     rates = pickle.load(f)

# # remove ('lomax', *, x>25, *, y<1/50)
# rates = {k: v for k, v in rates.items() if not (k[2] > 25 and 50 < k[4] < 1e6)}

with open("./scripts/tempa.txt", "r") as f:
    lines = f.readlines()

rates = {}
for line in lines:
    if line.startswith("('l") and ") " in line and (not line.endswith(")")):
        # split string to get the key
        strings = line.split(") ")
        key = eval(strings[0] + ")")
        v = strings[1]
        try:
            value = eval(strings[1])
        except:
            value = None
        rates[key] = value


with open(ANALYTICAL_FORK_RATES_PATH_STD, "wb") as f:
    pickle.dump(rates, f)
