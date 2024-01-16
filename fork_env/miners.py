from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Dlt:
    """
    A distributed ledger technology characterized by its block time
    """
    def __init__(self, miners: dict[int, Miner], block_propagation_time: float):
        self.miners: dict[int, Miner] = miners
        self.block_propagation_time: float = block_propagation_time
        self.last_mining_time: float | None = None

    def fork_created(self) -> bool:
        """
        run mine_block iteratively with each miner
        """
        for miner in self.miners.values():
            miner.mine_first_block()
        # sort mining times
        sorted_mining_times = sorted([miner.last_mining_time for miner in self.miners.values() if miner.last_mining_time is not None])
        time_diff = sorted_mining_times[1] - sorted_mining_times[0]
        self.last_mining_time = sorted_mining_times[0]
        if time_diff < self.block_propagation_time:
            print("Fork occurs")
            return True
        else:
            return False

@dataclass
class Miner:
    """
    A miner characterized by its id and hash rate
    """
    id: int
    hash_rate: float
    last_mining_time: float | None = None

    def mine_first_block(self):
        """
        Mine a block and add it to the dlt
        """
        # simulate a poisson arrival time
        mining_time = np.random.exponential(1 / self.hash_rate)
        self.last_mining_time = mining_time
