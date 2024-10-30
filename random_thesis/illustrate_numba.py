from numba import njit 
import time

# Make timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__}() ran in {time.time() - time_start:.6f}s')
        return result        
    return wrapper

# Normal double loop function
@timer
def f_1():
    for i in range(int(1e5)):
        for j in range(int(1e4)):
            pass
    return None

# Speed up with Numba - big increase in iterations
@timer
@njit
def f_2():
    try:
        for i in range(int(1e8)):
            for j in range(int(1e8)):
                pass
    except Exception:
        print("something went wrong...", Exception)
    return None

# Call functions
f_1()
f_2()

"""
f_1() ran in 10.036779s
f_2() ran in 0.380416s

Next: Try to combine this with Multiprocessing
"""