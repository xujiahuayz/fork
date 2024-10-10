import requests
from fork_env.constants import DATA_FOLDER

# URL of the file
url = "https://www.dsn.kastel.kit.edu/bitcoin/data/invstat.gpd"

# Fetch the data from the URL
response = requests.get(url)

# Save the content to a file
with open(DATA_FOLDER / "invstat.gpd", "wb") as file:
    file.write(response.content)
