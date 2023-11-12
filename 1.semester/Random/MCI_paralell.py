import numpy as np
import time
import multiprocessing as mp

"""
This code is not really good at what it was meant to do. 
I think 1D-integrals are not big enough (8D is limit where MC
is superior to other techniques). 
To split the domain into smaller pieces for multivariable integrals 
is hard, as the integral no longer is linear, but contain cross terms. 

1D: I = I_a + I_b
2D: I_x*I_y = (I_xa + I_xb)*(I_ya + I_yb)

which is a mess.
"""

x0 = 0
x1 = 1000
Nx = int(1e5)

proc_num = 4

N = 1000
step = (x1 - x0) / N

x = []
for i in range(N):
    xi = np.linspace(i*step, (i+1)*step, Nx)
    x.append(xi)

t0 = time.time()

def monte_carlo(func, x):
    x0 = x[0]
    x1 = x[-1]
    Nx = len(x)
    randx = np.random.uniform(x0, x1, Nx)
    W = f(randx)
    V = (x1 - x0)
    MCI = V * np.mean(W)          # Uniform so that probability = area_graph / volume
    return MCI

def f(x):
    return np.sin(x)


# I = monte_carlo(f, x)
# print(I)

config = []
for i in range(N):
    config.append([f, x[i]])

if __name__ == '__main__':

    with mp.Pool(proc_num) as pool:
        mc = pool.starmap(monte_carlo, config)      # Running on 6 threads repeating M times
    mc = np.array(mc)
    I = np.sum(mc)

    t1 = time.time()
    print(f'Computation time {t1 - t0}s')
    print(f'MC-integral: {I}')