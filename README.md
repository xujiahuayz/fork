# Fork

## Setup

```
git clone https://github.com/xujiahuayz/fork.git
cd fork
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

## Create data folder

```
mkdir data
```

## Download pool data to data folder

```
cd data
wget https://raw.githubusercontent.com/blockchain/Blockchain-Known-Pools/master/pools.json
wget https://raw.githubusercontent.com/bitcoin-data/stale-blocks/refs/heads/master/stale-blocks.csv
<!-- wget https://api.blockchain.info/charts/n-orphaned-blocks?timespan=all&sampled=true&metadata=false&daysAverageString=1d&cors=true&format=json -->
curl -H "authority: www.antpool.com" "https://www.antpool.com/api/v3/minerInfo/miner/list/all" > miner_rank.json
wget https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv
<!-- curl "https://api.blockchain.info/charts/n-orphaned-blocks?timespan=all&sampled=true&metadata=false&daysAverageString=1d&cors=true&format=json" -> "orphans.json" -->
```

## Run scripts

### get frequency fitting parameters

```zsh
python scripts/get_miner_freq.py
```

### generate calibrated historical fork frequency date

```zsh
python scripts/get_fork.py
```

### build the hash panel data table

```zsh
python scripts/build_hash_panel.py
```

### tabulate the parameter values

```zsh
python scripts/tabulate_hash.py
```

### plot ccdf

```zsh
python scripts/plot_miner_freq.py
```

### plot ccdf with zero miners

```zsh
python scripts/plot_miner_freq_zero.py
```

## Set up the environmental variables

put your APIs in `.env`:

```
CLOVERPOOL_API_1="t123"
CLOVERPOOL_API_2='t456'
```

```
export $(cat .env | xargs)
```

<!-- https://api.blockchain.info/charts/difficulty?timespan=all -->
