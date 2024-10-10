import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get x from 1 to 100

x = np.linspace(0.000001, 10, 1000000)

beta = 5
y_exp = np.exp(-beta * x)

alpha = 2

y = x ** (-alpha)

# plot

plt.plot(x, y, label="Power Law")
plt.plot(x, y_exp, label="Exponential")

# verticle line at 1/beta

plt.axvline(1 / beta, color="red", linestyle="--", label="1/beta")

# horizontal line at 1

plt.axhline(np.exp(-1), color="green", linestyle="--", label="1")

plt.xlabel("x")
plt.ylabel("y")
# log log scale
plt.xscale("log")
plt.yscale("log")

plt.title("Power Law and Exponential Distributions")
plt.legend()
