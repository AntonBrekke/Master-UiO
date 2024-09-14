import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt 
import sys

a = [0, 0, 0]     # Start points for variables x1, x2, ...
b = [np.pi, np.pi, np.pi]   # End points for variables x1, x2, ...
ni = int(1e3)        
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
    s = 1
    for n, x in enumerate(args, start=1):
        s *= np.exp(-1/n*x**2)
    return s

t0 = time.time()
I = monte_carlo(f, x)
t1 = time.time()
print(f'Computation time {t1 - t0}')
print(f'MC: {I}')

M = int(1e5)
mc = np.zeros(M)

if len(sys.argv) != 2:
    plot_hist = ''
else:
    plot_hist = sys.argv[1]

if plot_hist == 'hist':
    t0 = time.time()
    for m in range(M):
        I = monte_carlo(f, x)
        t1 = time.time()
        mc[m] = I
    t1 = time.time()
    
    print(f'Computation time {t1 - t0}s')

    # Making figure of data
    fig = plt.figure(facecolor='k')
    ax = fig.add_subplot(facecolor='k')
    ax.spines['bottom'].set_color('w')      # Setting axis white
    ax.spines['left'].set_color('w')
    ax.tick_params(axis='x', colors='w')    # Setting ticks white
    ax.tick_params(axis='y', colors='w')

    # Getting color-chart for histogram
    get_cmap = plt.get_cmap('jet')
    n, bins, patches = ax.hist(mc, bins=75)
    cmap = np.array(get_cmap(n / np.max(n)))    # Must normalize data

    for color, p in zip(cmap, patches):
        plt.setp(p, 'facecolor', color)

    plt.show()
