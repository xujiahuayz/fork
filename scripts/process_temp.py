import pickle

from fork_env.constants import DATA_FOLDER, SUM_HASH_RATE


with open(DATA_FOLDER / "per_c.pkl", "rb") as f:
    rates = pickle.load(f)

# remove ('lomax', *, x>25, *, y<1/50)
rates = {k: v for k, v in rates.items() if not (k[2] > 25 and 50 < k[4] < 1e6)}

with open("./scripts/tempa.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("('lomax', ") and ") " in line:
        # split string to get the key
        strings = line.split(") ")
        key = eval(strings[0] + ")")
        value = eval(strings[1])
        rates[key] = value


with open(DATA_FOLDER / "per_c_new.pkl", "wb") as f:
    pickle.dump(rates, f)
