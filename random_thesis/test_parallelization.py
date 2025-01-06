import numpy as np
from multiprocessing import Pool, cpu_count
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(x):
    return x**2

x = np.linspace(0, 5, int(1e3))

num_cpus = cpu_count()
num_process = int(2.5*num_cpus) # cpu_count(), 48

if __name__ == '__main__':
    info('main line')
    with Pool(num_process) as pool:
        result = pool.map(f, x, chunksize=1)