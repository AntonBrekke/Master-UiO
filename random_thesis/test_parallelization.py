import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time

"""
Mainly just to play around with some simple examples 
to see the effects of changing
* num_processes
* chunksize 
and try to figure out what works and what does not. 
In general: 
Say you have 12 tasks to do, each takes 1 minute, and 6 workers to complete the task. 
Then chunksize = 6 is really bad, as two workers ends up doing one task 6 times each
spending 6 minutes in total (doing all the work), while the others do nothing (like in kommunen).

In this case, chunksize = 1 is good, as all workers are busy only two times. 
Thus, a speed-up by 3 times. However, some additional time is added in between the 
two rounds of work, to allocate memory etc.   
"""

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(x):
    p = np.random.choice([0, 1, 2])
    if p == 1:
        time.sleep(0.2)
    if p == 2:
        time.sleep(0.3)
    return 1

Nx = int(2e3)
x = np.linspace(0, 1, Nx)

num_cpus = cpu_count()
num_process = int(4*num_cpus)

if __name__ == '__main__':
    print(f'num_cpus: {num_cpus}')
    print(f'num_processes: {num_process}')
    print(f'divide task: {int(Nx/num_process)} points')
    info('main line')
    import time 
    start = time.time()
    with Pool(num_process) as pool:
        result = pool.map(f, x, chunksize=1)
    end = time.time()
    # print(result)
    print(f'Ran in {end-start:.3f}s')