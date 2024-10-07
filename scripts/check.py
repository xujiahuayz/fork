from fork_env.constants import SUM_HASH_RATE, N_MINER, HASH_STD

# import gamma function
from scipy.special import gamma

hash_var = HASH_STD**2
hash_mean = SUM_HASH_RATE / N_MINER

alpha = 1 - hash_mean**2 / hash_var
ell = gamma(2 - alpha) / gamma(1 - alpha) / hash_mean

scaling_c = ell ** (1 - alpha) / gamma(1 - alpha)

# plot the distribution
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * hash_mean, 1000)
y = scaling_c * x ** (-alpha) * np.exp(-ell * x)

plt.plot(x, y)

# https://en.wikipedia.org/wiki/Power_law#Power_law_with_exponential_cutoff
