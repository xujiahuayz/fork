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
wget https://api.blockchain.info/charts/difficulty?timespan=all
curl -H "authority: www.antpool.com" "https://www.antpool.com/api/v3/minerInfo/miner/list/all" > miner_rank.json
```

## Run scripts

### get frequency fitting parameters

```zsh
python scripts/get_miner_freq.py
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

<!-- https://api.blockchain.info/charts/difficulty?timespan=all -->
