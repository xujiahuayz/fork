# open DATA_FOLDER / "miner_rank.json"

import json
import pandas as pd
from datetime import datetime

from fork_env.constants import DATA_FOLDER

with open(DATA_FOLDER / "miner_rank.json", "r") as f:
    miner_rank = json.load(f)

bitcoin_miners = miner_rank["data"]["items"][0]["minerModelList"]
miner_hashrate = [w["hashrateNumber"] for w in bitcoin_miners if w["coinType"] == "BTC"]

# sort the list miner_hashrate
miner_hashrate.sort(reverse=True)

# histogram of hashrate
miner_hashrate_df = pd.DataFrame(miner_hashrate, columns=["hashrate"])
miner_hashrate_df["hashrate"].plot.hist(bins=11)
