import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 10000)

q = np.linspace(0, 1, 10000)
W = np.where(
    q <= 0.5,
    6 * (q**3 - q**2) + 1,
    2 * (1 - q)**3
)



plt.plot(q, W)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()