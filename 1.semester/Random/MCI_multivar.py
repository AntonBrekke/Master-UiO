import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt 

a = [0, 0, 0]     # Start points for variables x1, x2, ...
b = [np.pi, np.pi, np.pi]   # End points for variables x1, x2, ...
ni = int(1e6)        
N = 3*[ni]      # Num of points in array for each variable 

x = []          # Collection of arrays for each variable 
for i in range(len(a)):
    x.append(np.linspace(a[i], b[i], N[i]))

def monte_carlo(func, args):
    V = 1
    randargs = []
    for xi in args:
        xi0 = xi[0]
        xi1 = xi[-1]
        Nxi = len(xi)
        randxi = np.random.uniform(xi0, xi1, Nxi)
        V *= (xi1 - xi0)
        randargs.append(randxi)
    W = func(randargs)
    MCI = V * np.mean(W)          
    return MCI

# Any arbitrary function, just made a choice
def f(args):
    x, y, z = args
    return z*np.sin(y)*np.exp(-x**2)

t0 = time.time()
I = monte_carlo(f, x)
t1 = time.time()

print(f'Computation time {t1 - t0}s')
print(f'MC: {I}')


