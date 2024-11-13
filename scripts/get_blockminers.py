# import requests
import time
import pandas as pd
import http.client
from datetime import timedelta, datetime
import json
from fork_env.settings import CLOVERPOOL_API_1, CLOVERPOOL_API_2


from fork_env.constants import DATA_FOLDER

dates = ["2009-01-03", "2024-05-11", "2024-11-10"]


full_list = []


# iterate through the required configurations
for i, api in enumerate([CLOVERPOOL_API_1, CLOVERPOOL_API_2]):
    print(f"Processing API {i + 1}")

    headers = {"X-API-TOKEN": api}

    current_date = datetime.strptime(dates[i], "%Y-%m-%d")
    while current_date <= datetime.strptime(dates[i + 1], "%Y-%m-%d"):
        # Format the date to match the API requirement (YYYYMMDD)
        date_str = current_date.strftime("%Y%m%d")

        conn = http.client.HTTPSConnection("tools-gateway.api.cloverpool.com")

        # send data pull request. API Documentation: https://tools.api.cloverpool.com/docs/en/#get-block-list-by-daytime
        conn.request(
            "GET", f"/rest/api/v1.0/nodeapi/block/date/{date_str}", headers=headers
        )

        data = conn.getresponse().read()

        # sleep for 2 seconds to avoid rate limiting
        time.sleep(2)

        day_data = json.loads(data.decode("utf-8"))

        assert (
            day_data.get("code") == 0 and "data" in day_data
        ), f"Error in fetching data for {date_str}"

        full_list.extend(day_data["data"])

        # break loop

        print(f"Processed {date_str}")

        current_date += timedelta(days=1)


block_miners_df = pd.DataFrame(full_list)
block_miners_df.to_pickle(DATA_FOLDER / "block_miners.pkl")
