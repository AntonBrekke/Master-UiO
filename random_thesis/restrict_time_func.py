from pebble import ProcessPool, ProcessExpired
from multiprocessing import cpu_count
from concurrent.futures import TimeoutError
import time 
import numpy as np


def function(n):
    time.sleep(n)
    return n

elements = np.linspace(0, 10, 11)

cpus = 6
chunksize = 1
# the timeout will be assigned to each chunk
# therefore, we need to consider its size
timeout = 5

if __name__ == '__main__':  

    with ProcessPool(max_workers=cpus) as pool:
        future = pool.map(function, elements, chunksize=chunksize, timeout=timeout)

        iterator = future.result()

        result_list = []

        while True:
            try:
                result = next(iterator)
                result_list.append(result)
            except TimeoutError as error:
                print(f"function took longer than {error} seconds")
            except StopIteration:
                break
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process

    # import matplotlib.pyplot as plt
    # print(result_arr.shape)
    print(result_list)
    # plt.plot(result_list[-1][1].t, result_list[-1][1].y[0])
    # plt.show()