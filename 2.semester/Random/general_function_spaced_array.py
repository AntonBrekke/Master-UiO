import numpy as np
import matplotlib.pyplot as plt 

# Function in which you want spacing,
# i.e logspace for log spacing.
# Samples points based on y-axis of function,
# weights where slope increase.
def f(x):
    return np.log(x)

# Inverse of function above.
def f_inv(x):
    return np.exp(x)

x0 = 2
x1 = 8
N = 50
xx = np.linspace(x0, x1, N)
# Sample from y-axis
x = np.linspace(f(x0), f(x1), N)
# Project y-axis point onto the x-axis
x_f = f_inv(x)

y = np.zeros(N)

yline = int(N/2)
plt.plot(x_f, f(x_f))
plt.plot(y + x0, x, 'go')
plt.plot(xx, y + f(5), 'k')
plt.plot(y + f_inv(f(5)), f(x_f), 'k')
plt.plot(x_f, y, 'ro')
plt.show()