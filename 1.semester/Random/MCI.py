import numpy as np
import time
from numba import njit 

# Start
x0 = 0
y0 = 0
z0 = 0
# End
x1 = np.pi
y1 = np.pi
z1 = np.pi

Nx = int(1e6)
Ny = int(1e6)
Nz = int(1e6)

t0 = time.time()

def monte_carlo(func):
    randx = np.random.uniform(x0, x1, Nx)
    randy = np.random.uniform(y0, y1, Ny)
    randz = np.random.uniform(z0, z1, Nz)
    W = f(randx, randy, randz)
    V = (x1 - x0)*(y1 - y0)*(z1 - z0)
    MCI = V * np.mean(W)          # Uniform so that probability = area_graph / volume
    return MCI

def f(x, y, z):
    return np.sin(x)*np.sin(y)*np.sin(z)      # Know answer is 8

I = monte_carlo(f)

t1 = time.time()
print(f'Computation time {t1 - t0}s')
print(I)

