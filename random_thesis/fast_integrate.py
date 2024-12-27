import numpy as np
from scipy.integrate import quad 
import numba as nb 
import time

"""
integral of 
sum_k=1^n int_0^\infty e^{-kx}/k = sum_k=0^n 1 / k^2 = zeta(2) = np.pi^2/6
"""

import warnings
warnings.filterwarnings('ignore')

def f_no_numba(x, c):
    return np.exp(-c*x) / c

@nb.njit
def f_numba(x, c):
    return np.exp(-c*x) / c

def integrate(func, a, b, arr):
    res = 0
    for i in range(len(arr)):
        I, err = quad(func, a, b, args=(arr[i]))
        res = res + I
    return res

@np.vectorize
def integrate_vec(func, a, b, arr):
    res = 0
    I, err = quad(func, a, b, args=(arr))
    res = res + I
    return res

n = int(1e4)
arr = np.array([*range(1, n+1)])

t1 = time.time()
I_no_numba = integrate(func=f_no_numba, a=0, b=np.inf, arr=arr)
t2 = time.time()

t3 = time.time()
I_numba = integrate(func=f_numba, a=0, b=np.inf, arr=arr)
t4 = time.time()

t5 = time.time()
I_numba_vec = np.sum(integrate_vec(func=f_numba, a=0, b=np.inf, arr=arr))
t6 = time.time()

print(I_no_numba, t2 - t1, 'no_numba')
print(I_numba, t4 - t3, 'numba')
print(I_numba_vec, t6 - t5, 'numba_vec')
print(np.pi**2/6, 'analytical')