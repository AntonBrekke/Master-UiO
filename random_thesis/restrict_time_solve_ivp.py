from pebble import ProcessPool, ProcessExpired
from multiprocessing import cpu_count
from concurrent.futures import TimeoutError

from scipy.integrate import solve_ivp
import numpy as np

"""
Kills processes that take too long to finish. 
Using "pebble" instead of "multiprocessing" because it has a timeout feature.
Does not seem to be completely stable for some reason.
"""

def RHS(t, y, k, w):
    return [np.sin(w*t)/(k*t)]

def sol_RHS(params):
    ti = 1
    tf = 10
    Nt = int(1e5)
    t = np.linspace(ti, tf, Nt)

    atol = 0.
    rtol = 1e-10
    max_step = 1.
    method = 'LSODA'

    w = params[0]
    k = params[1]

    y0 = np.array([1])
    sol = solve_ivp(fun=RHS, t_span=(ti, tf), args=(k, w), y0=y0, method=method, t_eval=t, events=[event_zero], atol=atol, rtol=rtol, max_step=max_step)
    return w, k, sol.success*10, w-k

def event_zero(t, y, k, w):
    return y


cpus = 6
chunksize = 1
timeout = 0.2
nw = 20
nk = 12
w_grid = np.linspace(0, 100, nw)
k_grid = np.linspace(1, 10, nk)

params_grid = np.array((np.repeat(w_grid, nk), np.tile(k_grid, nw))).T
# print(params_grid)

if __name__ == '__main__':  

    with ProcessPool(max_workers=cpus) as pool:
        future = pool.map(sol_RHS, params_grid, chunksize=chunksize, timeout=timeout)

        iterator = future.result()

        result_list = []

        iw = 0
        ik = 0
        openfile = open("./write_live_data_folder/live_data_solve_ivp.dat", "w")
        while True:
            openfile = open("./write_live_data_folder/live_data_solve_ivp.dat", "a")
            try:
                result = next(iterator)
                # result = np.array(result)
                result_list.append(result)
                openfile.write(str(result).strip('()').replace(',', '')+'\n')
            except StopIteration:
                break
            except TimeoutError as error:
                w = w_grid[iw]
                k = k_grid[ik]
                result = (w, k, np.nan, w-k)
                result_list.append(result)
                openfile.write(str(result).strip('()').replace(',', '')+'\n')
                print(f"function took longer than {timeout} seconds")
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process
            finally:
                ik += 1
                if ik == nk:
                    ik = 0
                    iw += 1
                openfile.close()

    import matplotlib.pyplot as plt
    # print(result_list)
    data = np.loadtxt("./write_live_data_folder/live_data_solve_ivp.dat")
    print(data)
    results = np.array(result_list).reshape(len(result_list), 4)
    params_final =  results[:,:-1]
    print(params_final)
    # np.savetxt(f'./write_live_data_folder/restrict_time_solve_ivp.dat', results)