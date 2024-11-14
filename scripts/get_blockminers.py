import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from fork_env.constants import BLOCK_MINER_CLOVERPOOL_PATH
from fork_env.settings import CLOVERPOOL_API_LIST

full_list = []

TODAY = datetime.today()

current_date = datetime.strptime("2009-01-03", "%Y-%m-%d")

for i, api in enumerate(CLOVERPOOL_API_LIST):
    print(f"Processing API {i + 1}")

    day_data = {"code": 0, "data": []}
    while (current_date <= TODAY) and (day_data["code"] == 0):
        full_list.extend(day_data["data"])
        # Format the date to match the API requirement (YYYYMMDD)
        date_str = current_date.strftime("%Y%m%d")
        day_data = requests.get(
            f"https://tools-gateway.api.cloverpool.com/rest/api/v1.0/nodeapi/block/date/{date_str}",
            headers={"X-API-TOKEN": api},
            timeout=10,
        ).json()

        if day_data["code"] == 0:
            print(f"Processed {date_str}")
            current_date += timedelta(days=1)
            time.sleep(0.83)
        else:
            print(f"Error in fetching data for {date_str}: {day_data}")


block_miners_df = pd.DataFrame(full_list)
block_miners_df.to_pickle(BLOCK_MINER_CLOVERPOOL_PATH)
