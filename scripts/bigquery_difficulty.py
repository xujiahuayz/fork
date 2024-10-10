import pickle
import os

# run `pip install --upgrade google-cloud-bigquery` in the terminal first
from google.cloud import bigquery

from fork_env.constants import DATA_FOLDER
from fork_env.settings import PROJECT_ROOT

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    PROJECT_ROOT / "gcloud_jiahuaxu_key.json"
)
client = bigquery.Client()

# Perform a query.
query_btc = """
SELECT number, bits FROM bigquery-public-data.crypto_bitcoin.blocks ORDER BY number
"""

query_job = client.query(query_btc)  # API request
rows = query_job.result()  # Waits for query to finish


field_names = [f.name for f in rows.schema]
# needs to be done in once, otherwise 'Iterator has already started' error
btc_tx_difficulty = [{field: row[field] for field in field_names} for row in rows]
# save the result to a pickle file
with open(DATA_FOLDER / "btc_tx_difficulty.pkl", "wb") as f:
    pickle.dump(btc_tx_difficulty, f)
