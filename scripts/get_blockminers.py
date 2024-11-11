# import requests
import pandas as pd
import http.client
from datetime import datetime, timedelta
import json

from fork_env.constants import DATA_FOLDER

conn = http.client.HTTPSConnection("tools-gateway.api.cloverpool.com")

api_configs = [
    {
        "api_key": "te756c5cbf84050f36b9bfe2561bc9c7d17629f4eb1e7058ed3dbfb3519238930",
        "start_date": "2009-01-03",
        "end_date": "2024-10-28",
        "output_file": "block_miners_1"
    },
    {
        "api_key": "t7939aef2fe7042c3d780497d3c957375c353efda88160665c6e8e034f7677320",
        "start_date": "2024-05-11",
        "end_date": "2024-10-28",
        "output_file": "block_miners_2"
    }
]

# iterate through the required configurations
for config in api_configs:
    conn = http.client.HTTPSConnection("tools-gateway.api.cloverpool.com")
    headers = {'X-API-TOKEN': config["api_key"]}


    full_list = []

    current_date = config["start_date"]
    while current_date <= config["end_date"]:
        # Format the date to match the API requirement (YYYYMMDD)
        date_str = current_date.strftime("%Y%m%d")

        # send data pull request. API Documentation: https://tools.api.cloverpool.com/docs/en/#get-block-list-by-daytime
        conn.request("GET", f"/rest/api/v1.0/nodeapi/block/date/{date_str}", headers=headers)

        res = conn.getresponse()
        data = res.read()

        day_data = json.loads(data.decode("utf-8"))

        if day_data.get("code") == 0 and "data" in day_data:
            # Collect only necessary fields (3 in our case)
            for entry in day_data["data"]:
                full_list.append({
                    "height": entry["height"],
                    "blocktime": datetime.fromtimestamp(entry["time"]),
                    "miner_cluster": entry["extras"]["minerName"]
                })
            # print(f"data pull for {current_date}")

        current_date += timedelta(days=1)

    block_miners_df = pd.DataFrame(full_list)

    block_miners_df.to_pickle(DATA_FOLDER / f"{config["output_file"]}.pkl")
    block_miners_df.to_csv(DATA_FOLDER / f"{config["output_file"]}.csv")

