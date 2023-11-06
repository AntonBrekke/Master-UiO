import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

x0 = 0
x1 = np.pi

y0 = 0
y1 = np.pi

Nx = int(1e6)
Ny = int(1e6)

x = np.linspace(x0, x1, Nx)
y = np.linspace(y0, y1, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

t0 = time.time()

def monte_carlo(func):
    randx = np.random.uniform(x0, x1, Nx)
    randy = np.random.uniform(x0, y1, Ny)
    W = f(randx, randy)
    area = (x1 - x0)*(y1 - y0)
    MCI = area * np.mean(W)          # Uniform so that probability = area_graph / area
    return MCI

def f(x, y):
    return np.sin(x)*np.sin(y)      # Know answer is 4

I = monte_carlo(f)

t1 = time.time()
print(f'Computation time {t1 - t0}s')
print(I)

