import numpy as np


# Define the custom joint PDF function
def joint_pdf(x, y):
    return np.exp(-x - y)  # example: product of two exponential PDFs


# Define the bounds for sampling
x_min, x_max = 0, np.inf
y_min, y_max = 0, np.inf

# Number of samples
n_samples = 1000

# Storage for samples
samples = []

# Rejection sampling
while len(samples) < n_samples:
    # Generate candidates from an exponential distribution
    x_candidate = np.random.exponential(scale=1.0)
    y_candidate = np.random.exponential(scale=1.0)
    u = np.random.uniform(0, 1)
    if u < joint_pdf(x_candidate, y_candidate):
        samples.append([x_candidate, y_candidate])

samples = np.array(samples)
x_samples = samples[:, 0]
y_samples = samples[:, 1]

# Display some of the samples
print(x_samples[:10])
print(y_samples[:10])
