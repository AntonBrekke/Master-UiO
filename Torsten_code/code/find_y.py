
#! /usr/bin/env python3

import numpy as np
from multiprocessing import Pool, cpu_count

import constants_functions as cf
import sterile_caller
import time
import sys, os

r_m = 5.
m_a = 0.
k_d = 1.
k_a = 1.
k_phi = -1.
dof_d = 2.
dof_phi = 1.

# The mass is given in GeV, but is plotted in kev. Use 1e-6 * x GeV = x kev
n_m = 5         # 21
n_th = 5        # 81
# m_d_grid = np.logspace(-6, -3.7, n_m)
m_d_grid = 1e-6*np.logspace(1.7, 2.5, n_m)          # 1e-6*m_d = 1e-6 * GeV = keV, plotted values: (10^0 - 10^2) keV
sin2_2th_grid = np.logspace(-17, -15, n_th)
# m_d_grid = 1e-6*10**(np.array([1.7, 2.24, 1.15, 2.3, 1.5]))         # 1e-6*m_d = 1e-6 * GeV = keV, plotted values: (10^0 - 10^2) keV
# sin2_2th_grid = 10**(np.array([-16.0762, -15.6541, -16.5350, -15.6037, -16.2373]))
# m_d_grid = 1e-6*10**(np.array([1.7, 1.15, 2.3, 1.5]))         # 1e-6*m_d = 1e-6 * GeV = keV, plotted values: (10^0 - 10^2) keV
# sin2_2th_grid = 10**(np.array([-16.0762, -16.5350, -15.6037, -16.2373]))
# m_d_grid = 1e-6*10**(np.array([2.00, 2.23, 2.41, 2.3, 1.15]))         # 1e-6*m_d = 1e-6 * GeV = keV, plotted values: (10^0 - 10^2) keV
# sin2_2th_grid = 10**(np.array([-17.9334, -17.7420, -17.5984, -15.6037, -16.5345]))

num_cpus = cpu_count()
num_process = int(2.5*num_cpus) # cpu_count()

params_grid = np.array((np.repeat(m_d_grid, m_d_grid.size), np.tile(sin2_2th_grid, sin2_2th_grid.size))).T

spin_facs = True
off_shell = False

dirname = './sterile_res/'
i_max = 0       # 60
i_skip = 20
def find_y(params):
    m_d = params[0]
    sin2_2th = params[1]
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    m_phi = r_m*m_d

    O_d_h2_dw = cf.O_h2_dw(m_d, th)     # Anton: Omega_DM * h^2 from Dodelson-Widrow mechanism
    log_enhance_req = np.log(cf.omega_d0/O_d_h2_dw)
    y_cur = np.sqrt(log_enhance_req/(1e19*sin2_2th/m_phi)) # 1.65e16 roughly the scaling factor from parameter scans
    print(f'm_d: {m_d:.2e}, sin2_2th: {sin2_2th:2e}, O_d_h2_dw: {O_d_h2_dw:2e}, log_enchance_req: {log_enhance_req:.2f}, y_cur: {y_cur:.2e}')
    if log_enhance_req < 0.:
        print(f'log_enhance_req < 0 : {log_enhance_req:.2f}')
        return m_d, m_phi, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # print(m_d, sin2_2th, O_d_h2_dw, log_enhance_req, y_cur)
    O_d_h2_old = O_d_h2_dw
    y_old = 0.
    max_y = 1.
    min_y = 0.
    exception_num = 0 
    for i in range(i_max + 1):
        try:
            time1 = time.time()
            print("Running sterile_caller.call ")
            t_grid, T_SM_grid, T_nu_grid, ent_grid, hubble_grid, sf_grid, T_d_grid, xi_d_grid, xi_phi_grid, n_d_grid, n_phi_grid, C_therm_grid, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3, reached_integration_end = sterile_caller.call(m_d, m_phi, m_a, k_d, k_phi, k_a, dof_d, dof_phi, sin2_2th, y_cur, spin_facs=spin_facs, off_shell=off_shell)
            print(f"sterile_caller.call ran in {time.time()-time1}s")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("something went wrong... ", e, exc_type, fname, exc_tb.tb_lineno)
            exception_num += 1
            return m_d, m_phi, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        O_d_h2_cur = n_d_grid[-1]*m_d*cf.s0/(ent_grid[-1]*cf.rho_crit0_h2)
        # print(m_d, m_phi, sin2_2th, y_cur, O_d_h2_cur)
        if O_d_h2_cur > cf.omega_d0:
            max_y = min(y_cur, max_y)
        elif O_d_h2_cur < cf.omega_d0 and ((y_cur > y_old) == (O_d_h2_cur > O_d_h2_old) or i == 0):
            min_y = max(y_cur, min_y)
        if np.abs(O_d_h2_cur - cf.omega_d0)/cf.omega_d0 < 1e-2 or np.abs(max_y-min_y)/max_y < 1e-6:
            break
        elif ((y_cur > y_old) != (O_d_h2_cur > O_d_h2_old) and i > 0) or not reached_integration_end: # already in freeze-out regime, reduce y_cur to go back to pandemic
            y_old = y_cur
            O_d_h2_old = O_d_h2_cur
            max_y = min(y_cur, max_y)
            y_cur = 0.5*(min_y + max_y)
            print('freeze-out regime or very large abundance, reducing y ', m_d, sin2_2th, O_d_h2_dw, y_old, y_cur)
        elif i < i_max:
            log_enhance_cur = np.log(O_d_h2_cur/O_d_h2_dw)
            y_old = y_cur
            O_d_h2_old = O_d_h2_cur
            y_cur = min(np.sqrt(log_enhance_req/log_enhance_cur), 2.)*y_cur if log_enhance_cur > 0. else 2.*y_cur
            if y_cur >= max_y:
                y_cur = 0.5*(y_old + max_y)
            elif y_cur <= min_y:
                y_cur = 0.5*(y_old + min_y)
            if np.abs(y_old-y_cur)/y_old < 1e-6:
                y_cur = y_old
                break
            # print(m_d, sin2_2th, O_d_h2_dw, log_enhance_req, log_enhance_cur, y_old, y_cur)

    # print(f'Exceptions: {exception_num} ')
    print('find_y.py for-loop done ')
    filename = f'md_{m_d:.4e}_mphi_{m_phi:.4e}_sin22th_{sin2_2th:.4e}_y_{y_cur:.4e}.dat'
    np.savetxt(dirname + 'benchmark_pts/' + filename, np.column_stack((t_grid[::-i_skip][::-1], T_SM_grid[::-i_skip][::-1], T_nu_grid[::-i_skip][::-1], ent_grid[::-i_skip][::-1], hubble_grid[::-i_skip][::-1], sf_grid[::-i_skip][::-1], T_d_grid[::-i_skip][::-1], xi_d_grid[::-i_skip][::-1], xi_phi_grid[::-i_skip][::-1], n_d_grid[::-i_skip][::-1], n_phi_grid[::-i_skip][::-1], C_therm_grid[::-i_skip][::-1], n_d_grid[::-i_skip][::-1]*m_d*cf.s0/(ent_grid[::-i_skip][::-1]*cf.rho_crit0_h2))))

    therm_ratio = C_therm_grid / (3.*hubble_grid*n_d_grid)
    x_therm = m_d/T_nu_grid[np.argmax(therm_ratio > 1.)]
    x_d_therm = m_d/T_d_grid[np.argmax(therm_ratio > 1.)]
    therm_ratio_max = np.amax(therm_ratio)
    O_d_h2 = n_d_grid[-1]*m_d*cf.s0/(ent_grid[-1]*cf.rho_crit0_h2)
    # print(m_d, m_phi, sin2_2th, y_cur, O_d_h2, x_therm, x_d_therm, therm_ratio_max, fs_length)
    print(f'Returned y = {y_cur:.2e} for m_d = {m_d:.2e}, sin2_2th = {sin2_2th:.2e}')
    return m_d, m_phi, sin2_2th, y_cur, O_d_h2, x_therm, x_d_therm, therm_ratio_max, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3

"""
find_y returns 16 values, but 17 (16 above + O_d_h2_no_spin_stat) values is given in data and plotting-files given by Torsten???
"""

if __name__ == '__main__':
    time1 = time.time()
    print("Start process of find_y.py...")
    print(f"Number of CPUs: {num_cpus}")
    print(f"Number of processes: {num_process}")
    with Pool(processes=num_process) as pool:
        results = pool.map(find_y, params_grid, chunksize=1)

    dt = time.time() - time1
    print(f"find_y.py ran in {dt//60//60}h {dt//60%60}m {dt%60}s")
    np.savetxt(dirname+f'rm_{r_m:.2e}_y_relic_new.dat', results)

    """
    Anton: doing some testing on runtimes
    time = days*60*60*24 + hours*60*60 + min*60 + sec
    => days = time//60//60//24, hours = time//60//60%24, min = time//60%60 etc.

    n_m = 1
    n_th = 1
    i_max = 10
    t_gp_pd = 300
    find_y.py ran in 1012.4572186470032s
    (for only 1x1 grid search!!! Crazy slow...)
    Assuming linear time complexity, 100*100 grid would take 
    ~ 1012*100*100 = 1e7s = 117d + 3h + 6m + 40s ; 117d ~ 3 months + 24 days (!) 

    num_process = 30

    n_m = 21
    n_th = 81
    (19d, 22h, 10m, 12s)

    n_m = 1
    n_th = 1
    i_max = 1
    t_gp_pd = 100
    find_y.py ran in 0.0h 1.0m 22.613s

    n_m = 2
    n_th = 2
    i_max = 1
    t_gp_pd = 100
    find_y.py ran in 0.0h 2.0m 29.997s

    n_m = 3
    n_th = 3
    i_max = 1
    t_gp_pd = 100
    find_y.py ran in 0.0h 4.0m 43.297s

    n_m = 4
    n_th = 4
    i_max = 1
    t_gp_pd = 100
    find_y.py ran in 0.0h 4.0m 23.917s          paralell program probably kicks in efficiency 

    n_m = 5
    n_th = 5
    i_max = 1
    t_gp_pd = 100
    find_y.py ran in 0.0h 7.0m 49.486s

    n_m = 10
    n_th = 10
    i_max = 1
    t_gp_pd = 100
    find_y.py ran in 0.0h 26.0m 56.067s

    n_m = 10
    n_th = 10
    i_max = 10
    t_gp_pd = 100
    find_y.py ran in 2.0h 42.0m 38.207s

    n_m = 10
    n_th = 10
    i_max = 1
    t_gp_pd = 300
    find_y.py ran in 4.0h 9.0m 14.983s
    => n_m=21, n_th=81, i_max=60 : O(n)-> 5-6 months! 

    n_m = 2
    n_th = 2
    i_max = 5
    t_gp_pd = 100
    find_y.py ran in 0.0h 5.0m 30.8374080657959s

    n_m = 20
    n_th = 20
    i_max = 10
    t_gp_pd = 100
    find_y.py ran in 2.0h 19.0m 34.03615379333496s

    n_m = 20
    n_th = 80
    i_max = 5
    t_gp_pd = 300
    find_y.py ran in 8.0h 46.0m 38.839345932006836s

    
    n_m = 20         # 21
    n_th = 20
    i_max = 5
    t_gp_pd = 100
    find_y.py ran in 10.0h 50.0m 20.120605945587158s

    n_m = 15         # 21
    n_th = 30
    i_max = 20
    t_gp_pd = 100
    find_y.py ran in 34.0h 21.0m 58.66656732559204s

    Bottlenecks:
    line 413 pandemolator.py: solve_ivp()       (***)
    class TimeTempRelation: t_gp_pd too large
    """
