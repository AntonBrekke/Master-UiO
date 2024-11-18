#! /usr/bin/env python3

import numpy as np
from multiprocessing import Pool, cpu_count
import time
import sys, os

import constants_functions as cf
import sterile_caller

r_m = 5.
m_a = 0.
k_d = 1.
k_a = 1.
k_phi = -1.
dof_d = 2.
dof_phi = 1.

# n_m = 21
# n_th = 81
# m_d_grid = np.logspace(-6, -3.7, n_m)
# sin2_2th_grid = np.logspace(-17, -8, n_th)

# The mass is given in GeV, but is plotted in kev. Use 1e-6 * x GeV = x kev
n_m = 10         # 21
n_th = 10        # 81
m_d_grid = 1e-6*np.logspace(1.7, 2.5, n_m)          # 1e-6*m_d = 1e-6 * GeV = keV, plotted values: (10^0 - 10^2) keV
sin2_2th_grid = np.logspace(-17, -15, n_th)
# m_d_grid = 1e-6*10**(np.array([1.7, 2.24, 1.15, 2.3, 1.5]))         
# sin2_2th_grid = 10**(np.array([-16.0762, -15.6541, -16.5350, -15.6037, -16.2373]))
# m_d_grid = 1e-6*10**(np.array([1.7, 1.15, 2.3, 1.5]))         
# sin2_2th_grid = 10**(np.array([-16.0762, -16.5350, -15.6037, -16.2373]))
# m_d_grid = 1e-6*10**(np.array([2.00, 2.23, 2.41, 2.3, 1.15]))
# sin2_2th_grid = 10**(np.array([-17.9334, -17.7420, -17.5984, -15.6037, -16.5345]))

num_cpus = cpu_count()
num_process = int(2.5*num_cpus) # cpu_count(), 48

params_grid = np.array((np.repeat(m_d_grid, n_th), np.tile(sin2_2th_grid, n_m))).T

spin_facs = True
off_shell = False

dirname = './sterile_res/'
i_max = 0      # 60
i_skip = 20
def find_y(params):
    m_d = params[0]
    sin2_2th = params[1]
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    m_phi = r_m*m_d

    O_d_h2_dw = cf.O_h2_dw(m_d, th)         # Anton: Omega_DM * h^2 from Dodelson-Widrow mechanism
    log_enhance_req = np.log(cf.omega_d0/O_d_h2_dw)
    if log_enhance_req < 0.:
        return m_d, m_phi, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    y_cur = np.sqrt(log_enhance_req/(1e19*sin2_2th/m_phi)) # 1.65e16 roughly the scaling factor from parameter scans
    print(f'm_d: {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, O_d_g2_dw: {O_d_h2_dw:.2e}, log_enhance_req: {log_enhance_req:.2e}, y_cur: {y_cur:.2e}')
    O_d_h2_old = O_d_h2_dw
    y_old = 0.
    max_y = 1.
    min_y = 0.
    for i in range(i_max + 1):
        try:
            time1 = time.time()
            print("Running sterile_caller.call ")
            t_grid, T_SM_grid, T_nu_grid, ent_grid, hubble_grid, sf_grid, T_d_grid, xi_d_grid, xi_phi_grid, n_d_grid, n_phi_grid, C_therm_grid, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3, reached_integration_end = sterile_caller.call(m_d, m_phi, m_a, k_d, k_phi, k_a, dof_d, dof_phi, sin2_2th, y_cur, spin_facs = spin_facs, off_shell = off_shell)
            print(f"sterile_caller.call ran in {time.time()-time1}s")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("something went wrong... ", e, exc_type, fname, exc_tb.tb_lineno)
            return m_d, m_phi, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        O_d_h2_cur = n_d_grid[-1]*m_d*cf.s0/(ent_grid[-1]*cf.rho_crit0_h2)
        print(f'm_d: {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, y_cur: {y_cur:.2e}, O_d_g2_cur: {O_d_h2_cur:.2e}')
        if O_d_h2_cur > cf.omega_d0:
            max_y = min(y_cur, max_y)
        elif O_d_h2_cur < cf.omega_d0 and ((y_cur > y_old) == (O_d_h2_cur > O_d_h2_old) or i == 0):
            min_y = max(y_cur, min_y)
        if np.abs(O_d_h2_cur-cf.omega_d0)/cf.omega_d0 < 1e-2 or np.abs(max_y-min_y)/max_y < 1e-6:
            break
        elif ((y_cur > y_old) != (O_d_h2_cur > O_d_h2_old) and i > 0) or not reached_integration_end: # already in freeze-out regime, reduce y_cur to go back to pandemic
            y_old = y_cur
            O_d_h2_old = O_d_h2_cur
            max_y = min(y_cur, max_y)
            y_cur = 0.5*(min_y+max_y)
            print('freeze-out regime or very large abundance, reducing y', m_d, sin2_2th, O_d_h2_dw, y_old, y_cur)
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

    print('find_y.py for-loop done ')
    filename = f'md_{m_d:.4e}_mphi_{m_phi:.4e}_sin22th_{sin2_2th:.4e}_y_{y_cur:.4e}.dat'
    np.savetxt(dirname+'benchmark_pts/'+filename, np.column_stack((t_grid[::-i_skip][::-1], T_SM_grid[::-i_skip][::-1], T_nu_grid[::-i_skip][::-1], ent_grid[::-i_skip][::-1], hubble_grid[::-i_skip][::-1], sf_grid[::-i_skip][::-1], T_d_grid[::-i_skip][::-1], xi_d_grid[::-i_skip][::-1], xi_phi_grid[::-i_skip][::-1], n_d_grid[::-i_skip][::-1], n_phi_grid[::-i_skip][::-1], C_therm_grid[::-i_skip][::-1], n_d_grid[::-i_skip][::-1]*m_d*cf.s0/(ent_grid[::-i_skip][::-1]*cf.rho_crit0_h2))))

    therm_ratio = C_therm_grid / (3.*hubble_grid*n_d_grid)
    x_therm = m_d/T_nu_grid[np.argmax(therm_ratio > 1.)]
    x_d_therm = m_d/T_d_grid[np.argmax(therm_ratio > 1.)]
    therm_ratio_max = np.amax(therm_ratio)
    O_d_h2 = n_d_grid[-1]*m_d*cf.s0/(ent_grid[-1]*cf.rho_crit0_h2)
    print(f'Returned y = {y_cur:.2e} for m_d = {m_d:.2e}, sin2_2th = {sin2_2th:.2e}')
    return m_d, m_phi, sin2_2th, y_cur, O_d_h2, x_therm, x_d_therm, therm_ratio_max, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3

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
