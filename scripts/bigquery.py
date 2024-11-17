import gzip
import os
import pickle

# run `pip install --upgrade google-cloud-bigquery` in the terminal first
from google.cloud import bigquery

from fork_env.constants import BITCOIN_MINER_PATH
from fork_env.settings import PROJECT_ROOT

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    PROJECT_ROOT / "gcloud_jiahuaxu_key.json"
)
client = bigquery.Client()

# Perform a query.
query_btc = """
SELECT 
  block_number,
  block_timestamp, 
  output.addresses AS addresses
FROM 
  bigquery-public-data.crypto_bitcoin.transactions,
  UNNEST(outputs) AS output
WHERE 
  is_coinbase
ORDER BY 
  block_number
"""

query_job = client.query(query_btc)  # API request
rows = query_job.result()  # Waits for query to finish


field_names = [f.name for f in rows.schema]
# needs to be done in once, otherwise 'Iterator has already started' error
btc_tx_coinbase = [{field: row[field] for field in field_names} for row in rows]

# save result to a pickle file gzipped
with gzip.open(BITCOIN_MINER_PATH, "wb") as f:
    pickle.dump(btc_tx_coinbase, f)
