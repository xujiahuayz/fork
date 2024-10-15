import requests
from fork_env.constants import DATA_FOLDER
import pandas as pd


def gpd_to_pickle(
    url: str, filename: str, start_line: int, columns: list[str]
) -> pd.DataFrame:
    response = requests.get(url, timeout=99)
    rows = response.text.split("\n")[start_line:]
    df = pd.DataFrame(
        [row.split("\t") for row in rows],
        columns=columns,
    ).dropna()
    df.to_pickle(DATA_FOLDER / filename)
    return df


if __name__ == "__main__":

    invstat = gpd_to_pickle(
        url="https://www.dsn.kastel.kit.edu/bitcoin/data/invstat.gpd",
        filename="invstat.pkl",
        start_line=11,
        columns=[
            "start_unix",
            "end_unix",
            "total_inv",
            "tx50_title",
            "tx50",
            "tx90_title",
            "tx90",
            "tx99_title",
            "tx99",
            "block50_title",
            "block50",
            "block90_title",
            "block90",
            "block99_title",
            "block99",
            "start_time",
            "end_time",
        ],
    )

    forks = gpd_to_pickle(
        url="https://www.dsn.kastel.kit.edu/bitcoin/forks/forks.gpd",
        filename="forks.pkl",
        start_line=13,
        columns=[
            "block_number",
            "time_diff",
            "first_time_main",
            "first_time_orphan",
            "first_time_subsequent",
            "number_peers",
            "miner_main",
            "miner_orphan",
            "miner_subsequent",
        ],
    )
