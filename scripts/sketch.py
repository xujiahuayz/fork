from matplotlib import pyplot as plt


def func(bi: int, lbda: float) -> float:
    return bi - pow(bi, 2) / (2 * lbda) - lbda / 2


# plot the function with bi = 2007 and lbda = 1 to 30000
bi = 2007
lbda = range(1, 30000)
y = [func(bi, i) for i in lbda]
plt.plot(lbda, y)
# check monotonicity of the function
for i in range(len(y) - 1):
    if y[i] > y[i + 1]:
        print("Function is not monotonic")
        break
