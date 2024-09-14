import numpy as np
import matplotlib.pyplot as plt
import pynverse as pyinv  

# Function in which you want spacing,
# i.e logspace for log spacing.
# Samples points based on y-axis of function,
# weights where slope increase.
def f(x):
    return np.log(x)

# # Inverse of function above.
# def f_inv(x):
#     return np.exp(x)

# Or use library 
f_inv = pyinv.inversefunc(f)

x0 = 0.1
x1 = 1
N = 20
xx = np.linspace(x0, x1, N)
# Sample from y-axis
y = np.linspace(f(x0), f(x1), N)
# Project y-axis point onto the x-axis
x_f = f_inv(y)

Y = np.zeros(N)

yline = int(N/2)
plt.plot(x_f, f(x_f), marker='o')
plt.plot(Y + x0, y, 'go')
# plt.plot(xx, y + f(5), 'k')
# plt.plot(y + f_inv(f(5)), f(x_f), 'k')
plt.plot(x_f, Y, 'ro')
plt.show()