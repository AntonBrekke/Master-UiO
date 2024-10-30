#! /usr/bin/env python3

import numpy as np
import time
from math import cos, sin, asin, sqrt, exp, log
from scipy.integrate import solve_ivp

import constants_functions as cf
import utils
import densities as dens

import scalar_mediator
import pandemolator as pandemolator

# Anton: Fermi-constant, Z,W boson mass in GeV 
GF = 1.166378e-5
mZ = 91.1876
mW = 80.379

def call(m_d, m_phi, m_a, k_d, k_phi, k_a, dof_d, dof_phi, sin2_2th, y, spin_facs=True, off_shell=False):
    m_phi2 = m_phi*m_phi
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    c_th = np.cos(th)
    s_th = np.sin(th)
    y2 = y*y
    # print(th, c_th, s_th, y2)

    M2_dd = 2.*y2*(c_th**4.)*(m_phi2 - 4.*m_d*m_d)
    M2_aa = 2.*y2*(s_th**4.)*(m_phi2 - 4.*m_a*m_a)
    M2_da = 2.*y2*(s_th**2.)*(c_th**2.)*(m_phi2 - ((m_a + m_d)**2.))

    vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
    vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
    vert_el = y2*y2*(c_th**8.)

    Gamma_phi = scalar_mediator.Gamma_phi(y, th, m_phi, m_d)
    m_Gamma_phi2 = m_phi2*Gamma_phi*Gamma_phi

    if spin_facs:
        import C_res_scalar
        if m_phi > 2.*m_d:
            import C_res_scalar_no_spin_stat as C_res_scalar_no_spin_stat
            # n = n_d + 2.*n_phi
            def C_n(T_a, T_d, xi_d, xi_phi):
                if T_a < m_phi / 50.:
                    return 0.
                C_pp_dd = C_res_scalar.C_n_pp_dd(m_d, m_phi, k_d, k_phi, T_d, xi_d, xi_phi, vert_el) / 4. # symmetry factor 1/4
                if not off_shell:
                    C_da = C_res_scalar.C_n_3_12(m_d, m_a, m_phi, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d,   0., xi_phi, M2_da)
                    C_aa = 0.#C_res_scalar.C_n_3_12(m_a, m_a, m_phi, k_a, k_a, k_phi, T_a, T_a, T_d,   0.,   0., xi_phi, M2_aa) / 2.
                    C_da_dd = 0.
                    C_aa_dd = 0.
                else:
                    C_da = 0.
                    C_aa = 0.
                    C_da_dd = C_res_scalar.C_34_12(0, 1., -1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, res_sub=False, thermal_width=True) / 2.
                    C_aa_dd = 0.#C_res_scalar.C_34_12(0, 2., -2., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_phi2, m_Gamma_phi2, res_sub=False, thermal_width=True) / 4.
                return C_da + 2.*C_aa + C_da_dd + C_aa_dd + 2.*C_pp_dd
            # rho = rho_d + rho_phi
            def C_rho(T_a, T_d, xi_d, xi_phi):
                if T_a < m_phi / 50.:
                    return 0.
                if not off_shell:
                    C_da = C_res_scalar.C_rho_3_12(2, m_d, m_a, m_phi, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d, 0., xi_phi, M2_da)
                    C_aa = 0.#C_res_scalar.C_rho_3_12(3, m_a, m_a, m_phi, k_d, k_a, k_phi, T_a, T_a, T_d,   0., 0., xi_phi, M2_aa) / 2. # symmetry factor 1/2
                    C_da_dd = 0.
                    C_aa_dd = 0.
                else:
                    C_da = 0.
                    C_aa = 0.
                    C_da_dd = C_res_scalar.C_34_12(4, 1., -1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, res_sub=False, thermal_width=True) / 2.
                    C_aa_dd = 0.#C_res_scalar.C_34_12(12, 1., -1., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_phi2, m_Gamma_phi2, res_sub=False, thermal_width=True) / 4.
                return C_da + C_aa + C_da_dd + C_aa_dd
            
            def C_xi0(T_a, T_d, xi_d, xi_phi):
                if T_a < m_phi / 50.:
                    return 0.
                C_pp_dd = np.abs(C_res_scalar.C_n_pp_dd(m_d, m_phi, k_d, k_phi, T_d, xi_d, xi_phi, vert_el, type=1) / 4.)
                C_dd_pp = np.abs(C_res_scalar.C_n_pp_dd(m_d, m_phi, k_d, k_phi, T_d, xi_d, xi_phi, vert_el, type=-1) / 4.)
                return min(2.*C_pp_dd, 2.*C_dd_pp)
            
            def C_therm(T_d, xi_d, xi_phi):
                C_dd_p = C_res_scalar.C_n_3_12(m_d, m_d, m_phi, k_d, k_d, k_phi, T_d, T_d, T_d, xi_d, xi_d, xi_phi, M2_dd, type=+1) / 2.
                return 2.*C_dd_p
            
            def C_therm_kd(T_d, xi_d, xi_phi):
                if T_d > m_phi:
                    return 2. * C_res_scalar.C_n_3_12(m_d, m_d, m_phi, k_d, k_d, k_phi, T_d, T_d, T_d, xi_d, xi_d, xi_phi, M2_dd, type=+1) / 2.
                elif m_d / T_d - xi_d < 4.:
                    return 2. * C_res_scalar.C_34_12(0, 1., 0., m_d, m_d, m_d, m_d, k_d, k_d, k_d, k_d, T_d, T_d, T_d, T_d, xi_d, xi_d, xi_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False, thermal_width=True) / 4.
                return 2. * C_res_scalar_no_spin_stat.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False) / 4.
        else:
            print("Implementation needs to be updated... ")
            exit(1)
            import C_res_scalar_no_spin_stat as C_res_scalar_no_spin_stat
            # n = n_d + n_phi
            def C_n(T_a, T_d, xi_d, xi_phi):
                C_da = 0.# vanishes since net number change is zero
                C_aa = 0.#C_res_scalar.C_n_3_12(m_a, m_a, m_phi, k_a, k_a, k_phi, T_a, T_a, T_d,   0.,   0., xi_phi, M2_aa) / 2.
                C_da_dd = C_res_scalar.C_34_12(0, 1., -1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, res_sub=False) / 2.
                C_aa_da = 0.#C_res_scalar.C_34_12(0, 1., -1., m_d, m_a, m_a, m_a, k_d, k_a, k_a, k_a, T_d, T_a, T_a, T_a, xi_d, 0., 0., 0., vert_fi_1, m_phi2, m_Gamma_phi2, res_sub=False) / 2.
                C_aa_dd = 0.#C_res_scalar.C_34_12(0, 2., -2., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi_2, m_phi2, m_Gamma_phi2, res_sub=False) / 4.
                return C_da + C_aa + C_da_dd + C_aa_da + C_aa_dd
            # rho = rho_d + rho_phi
            def C_rho(T_a, T_d, xi_d, xi_phi):
                C_da = C_res_scalar.C_rho_3_12(2, m_d, m_a, m_phi, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d, 0., xi_phi, M2_da)
                C_aa = 0.#C_res_scalar.C_rho_3_12(3, m_a, m_a, m_phi, k_d, k_a, k_phi, T_a, T_a, T_d,   0., 0., xi_phi, M2_aa) / 2. # symmetry factor 1/2
                C_da_dd = C_res_scalar.C_34_12(4, 1., -1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, res_sub=False) / 2.
                C_aa_da = 0.#C_res_scalar.C_34_12(1, 1., -1., m_d, m_a, m_a, m_a, k_d, k_a, k_a, k_a, T_d, T_a, T_a, T_a, xi_d, 0., 0., 0., vert_fi_1, m_phi2, m_Gamma_phi2, res_sub=False) / 2.
                C_aa_dd = 0.#C_res_scalar.C_34_12(12, 1., -1., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi_2, m_phi2, m_Gamma_phi2, res_sub=False) / 4.
                return C_da + C_aa + C_da_dd + C_aa_da + C_aa_dd
            
            def C_xi0(T_a, T_d, xi_d, xi_phi):
                return 0.
            
            def C_therm(T_d, xi_d, xi_phi): # use collision operators without spin-stat. factors, only proxy here
                return 0.
                C_dd_dd = C_res_scalar_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, type=0, res_sub=False) / 4.
                C_pp_dd = C_res_scalar_no_spin_stat.C_pp_dd(m_d, m_phi, k_phi, T_d, xi_phi, vert_el, type=0) / 4.
                return min(2.*C_dd_dd, 2.*C_pp_dd)
            
            def C_therm_kd(T_d, xi_d, xi_phi):
                # C_dd_dd = C_res_scalar_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, type=0, res_sub=False) / 4.
                C_dd_dd = C_res_scalar_no_spin_stat.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False) / 4.
                return 2.*C_dd_dd
        # def G_d(T_d, xi_d, xi_phi):
        #     return C_res_scalar.Gamma_scat(T_d, m_d, m_d, m_phi, k_d, k_phi, T_d, T_d, xi_d, xi_phi, M2_dd)
    else:
        import C_res_scalar_no_spin_stat as C_res_scalar
        if m_phi > 2.*m_d:
            # n = n_d + 2.*n_phi
            def C_n(T_a, T_d, xi_d, xi_phi):
                if T_a < m_phi / 50.:
                    return 0.
                C_pp_dd = (- C_res_scalar.C_pp_dd(m_d, m_phi, k_phi, T_d, xi_phi, vert_el, type=0) + C_res_scalar.C_dd_pp(m_d, m_phi, k_d, T_d, xi_d, vert_el, type=0)) / 4. # symmetry factor 1/4
                if not off_shell:
                    C_da = C_res_scalar.C_12_3(m_d, m_a, m_phi, k_d, k_a, T_d, T_a, xi_d, 0., M2_da, type=0)
                    C_da_dd = 0.
                else:
                    C_da = 0.
                    C_da_dd = C_res_scalar.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=0, res_sub=False) / 2.
                return C_da + C_da_dd + 2.*C_pp_dd
            # rho = rho_d + rho_phi
            def C_rho(T_a, T_d, xi_d, xi_phi):
                if T_a < m_phi / 50.:
                    return 0.
                if not off_shell:
                    C_da = C_res_scalar.C_12_3(m_d, m_a, m_phi, k_d, k_a, T_d, T_a, xi_d, 0., M2_da, type=1)
                    C_da_dd = 0.
                else:
                    C_da = 0.
                    C_da_dd = C_res_scalar.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=1, res_sub=False) / 2.
                return C_da + C_da_dd
            
            def C_xi0(T_a, T_d, xi_d, xi_phi):
                if T_a < m_phi / 50.:
                    return 0.
                C_pp_dd = C_res_scalar.C_pp_dd(m_d, m_phi, k_phi, T_d, xi_phi, vert_el, type=0) / 4.
                return 2.*C_pp_dd
            
            def C_therm(T_d, xi_d, xi_phi):
                C_dd_p = C_res_scalar.C_12_3(m_d, m_d, m_phi, k_d, k_d, T_d, T_d, xi_d, xi_d, M2_dd, type = 0) / 2.
                return 2. * C_dd_p
            
            def C_therm_kd(T_d, xi_d, xi_phi):
                C_dd_dd = C_res_scalar.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False) / 4.
                return 2.*C_dd_dd
        else:
            print("Implementation needs to be updated... ")
            exit(1)
            # n = n_d + n_phi
            def C_n(T_a, T_d, xi_d, xi_phi):
                C_da = 0.# vanishes since net number change is zero
                C_da_dd = C_res_scalar.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=0, res_sub=False) / 2.
                return C_da_dd
            # rho = rho_d + rho_phi
            def C_rho(T_a, T_d, xi_d, xi_phi):
                C_da = C_res_scalar.C_12_3(m_d, m_a, m_phi, k_d, k_a, T_d, T_a, xi_d, 0., M2_da, type=1)
                C_da_dd = C_res_scalar.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=1, res_sub=False) / 2.
                return C_da + C_da_dd
            def C_xi0(T_a, T_d, xi_d, xi_phi):
                return 0.
            def C_therm(T_d, xi_d, xi_phi):
                return 0.
                # return C_res_scalar.C_12_3(m_d, m_a, m_phi, k_d, k_a, T_d, T_a, xi_d, 0., M2_da, type=0)
                C_dd_dd = C_res_scalar.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, type=0, res_sub=False) / 4.
                C_pp_dd = C_res_scalar.C_pp_dd(m_d, m_phi, k_phi, T_d, xi_phi, vert_el, type=0) / 4.
                return [2.*C_dd_dd, 2.*C_pp_dd]
            def C_therm_kd(T_d, xi_d, xi_phi):
                C_dd_dd = C_res_scalar.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False) / 4.
                return 2.*C_dd_dd
        # def G_d(T_d, xi_d, xi_phi):
        #     return C_res_scalar.Gamma_scat(T_d, m_d, m_d, m_phi, k_d, T_d, xi_phi, M2_dd)

    Ttrel = pandemolator.TimeTempRelation()
    ent_grid = np.array([cf.s_SM_no_nu(T) + cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
    sf_norm_today = (cf.s0/ent_grid)**(1./3.)
    T_d_dw = cf.T_d_dw(m_d) # temperature of maximal d production by Dodelson-Widrow mechanism
    i_ic = np.argmax(Ttrel.T_nu_grid < T_d_dw)
    i_end = np.argmax(Ttrel.T_nu_grid < m_d/2e1)
    sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
    n_ic = cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)
    rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0

    time1 = time.time()
    pan = pandemolator.Pandemolator(m_d, k_d, dof_d, m_phi, k_phi, dof_phi, m_a, k_a, C_n, C_rho, C_xi0, Ttrel.t_grid, Ttrel.T_nu_grid, Ttrel.dTnu_dt_grid, ent_grid, Ttrel.hubble_grid, Ttrel.sf_grid, i_ic, n_ic, rho_ic, i_end)
    time1 = time.time()
    print("Running Pandemolator.pandemolate ")
    pan.pandemolate()
    print(f"Pandemolator.pandemolate ran in {time.time() - time1}s ")

    try:
        C_therm_grid = np.array([C_therm(T_d, xi_d, xi_phi) for T_d, xi_d, xi_phi in zip(pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol)])
    except:
        C_therm_grid = np.zeros(pan.T_chi_grid_sol.size)

    O_d_h2 = pan.n_chi_grid_sol[-1]*m_d*cf.s0/(ent_grid[pan.i_end]*cf.rho_crit0_h2)

    if pan.T_chi_grid_sol.size < i_end - i_ic + 1: # integration of ode was stopped (abundance too large); issues calculating fs_length
        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol, pan.n_chi_grid_sol, pan.n_phi_grid_sol, C_therm_grid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False
    elif O_d_h2 < 1e-2*cf.omega_d0 or O_d_h2 > 1e1*cf.omega_d0: # computation of lambda_fs does not work for very small O_d_h2
        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol, pan.n_chi_grid_sol, pan.n_phi_grid_sol, C_therm_grid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True

    try:
        T_d_grid = np.empty(Ttrel.t_grid.size - i_ic)
        xi_d_grid = np.empty(Ttrel.t_grid.size - i_ic)
        xi_phi_grid = np.empty(Ttrel.t_grid.size - i_ic)
        n_d_grid = np.empty(Ttrel.t_grid.size - i_ic)
        n_phi_grid = np.empty(Ttrel.t_grid.size - i_ic)
        P_grid = np.empty(Ttrel.t_grid.size - i_ic)
        rho_grid = np.empty(Ttrel.t_grid.size - i_ic)

        n_sol = pan.T_chi_grid_sol.size
        T_d_grid[:n_sol] = pan.T_chi_grid_sol
        xi_d_grid[:n_sol] = pan.xi_chi_grid_sol
        xi_phi_grid[:n_sol] = pan.xi_phi_grid_sol
        n_d_grid[:n_sol] = pan.n_chi_grid_sol
        n_phi_grid[:n_sol] = pan.n_phi_grid_sol
        P_grid[:n_sol] = np.array([pan.P(T_d, xi_d, xi_phi) for T_d, xi_d, xi_phi in zip(T_d_grid[:n_sol], xi_d_grid[:n_sol], xi_phi_grid[:n_sol])])
        rho_grid[:n_sol] = np.array([pan.rho(T_d, xi_d, xi_phi) for T_d, xi_d, xi_phi in zip(T_d_grid[:n_sol], xi_d_grid[:n_sol], xi_phi_grid[:n_sol])])

        if n_sol < T_d_grid.size:
            H_interp = utils.LogInterp(Ttrel.t_grid, Ttrel.hubble_grid)
            sf_grid_tmp = (ent_grid[i_ic + n_sol - 1]/ent_grid)**(1./3.)
            sf_interp = utils.LogInterp(Ttrel.t_grid, sf_grid_tmp)
            def der(log_t, y):
                t = exp(log_t)
                H = H_interp(t) if t < Ttrel.t_grid[-1] else Ttrel.hubble_grid[-1]
                sf = sf_interp(t) if t < Ttrel.t_grid[-1] else sf_grid_tmp[-1]
                T = y[0]/(sf*sf)
                xi_diff = y[1]
                x = T/m_d
                x2 = x*x
                x3 = x2*x
                x4 = x3*x
                x5 = x4*x
                x6 = x5*x

                T_der = t*sf*sf*H*T*(x*5. - x2*20. + x3*595./8. - x4*2125/8. + x5*117475./128. - x6*49025./16.)
                xi_diff_der = t*H*(-x*15./4. + x2*135./8. - x3*4005./64. + x4*28035./128. - x5*378675./512. + x6*2463075./1024.)

                return [T_der, xi_diff_der]
            sol = solve_ivp(der, [log(Ttrel.t_grid[n_sol+i_ic-1]), log(Ttrel.t_grid[-1])], [T_d_grid[n_sol-1], xi_d_grid[n_sol-1]-m_d/T_d_grid[n_sol-1]], t_eval=np.log(Ttrel.t_grid[n_sol+i_ic:]), rtol=1e-6, atol=0., method='RK45', max_step=1.)
            T_d_grid[n_sol:] = sol.y[0, :] / (sf_grid_tmp[n_sol+i_ic:]**2.)
            xi_d_grid[n_sol:] = sol.y[1, :] + m_d / T_d_grid[n_sol:]
            xi_phi_grid[n_sol:] = 2.*xi_d_grid[n_sol:]
            n_d_grid[n_sol:] = n_d_grid[n_sol - 1] * ent_grid[i_ic + n_sol:] / ent_grid[i_ic + n_sol - 1]
            n_phi_grid[n_sol:] = 0.
            P_grid[n_sol:] = T_d_grid[n_sol:] * n_d_grid[n_sol:]
            rho_grid[n_sol:] = (m_d + 1.5*T_d_grid[n_sol:]) * n_d_grid[n_sol:]

        dPdt_grid = utils.fpsder(Ttrel.t_grid[i_ic:], P_grid)
        drhodt_grid = utils.fpsder(Ttrel.t_grid[i_ic:], rho_grid)
        mask = (dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.3)
        c_sound_grid = np.zeros(T_d_grid.size)
        c_sound_grid[mask] = np.sqrt(dPdt_grid[mask] / drhodt_grid[mask])
        # c_sound_all = np.zeros(T_d_grid.size)
        # c_sound_all[dPdt_grid/drhodt_grid > 0.] = np.sqrt(dPdt_grid[dPdt_grid/drhodt_grid > 0.] / drhodt_grid[dPdt_grid/drhodt_grid > 0.])
        # c_sound_no_pan = np.zeros(T_d_grid.size)
        # c_sound_no_pan[(dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.34)] = np.sqrt(dPdt_grid[(dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.34)] / drhodt_grid[(dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.34)])
        # # dPdrho = np.diff(P_grid)/np.diff(rho_grid)
        # import matplotlib.pyplot as plt
        # plt.loglog(m_d / Ttrel.T_nu_grid[i_ic:], c_sound_all)
        # plt.loglog(m_d / Ttrel.T_nu_grid[i_ic:], c_sound_no_pan)
        # plt.loglog(m_d / Ttrel.T_nu_grid[i_ic:], c_sound_grid)
        # plt.xlabel('1e-16')
        # plt.show()

        i_kd = np.argmax(Ttrel.T_nu_grid[i_ic:] < 0.1*m_phi)
        found_kd_coarse = False
        i_kd_start = i_kd
        C_therm_kd_last = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_phi_grid[i_kd])
        T_d_last = T_d_grid[i_kd]
        while i_kd < Ttrel.t_grid.size - i_ic - 1:
            try:
                C_therm_kd_cur = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_phi_grid[i_kd])
            except:
                C_therm_kd_cur = 0.
            if C_therm_kd_cur <= 0.:
                C_therm_kd_cur = C_therm_kd_last * ((T_d_grid[i_kd]/T_d_last)**3.5)
            C_therm_kd_last = C_therm_kd_cur
            T_d_last = T_d_grid[i_kd]
            ratio_kd = C_therm_kd_cur / (3.*Ttrel.hubble_grid[i_ic+i_kd]*n_d_grid[i_kd])
            # print('kd', m_d/T_d_grid[i_kd], m_d/T_d_grid[i_kd] - xi_d_grid[i_kd], ratio_kd)
            if ratio_kd < 1. and found_kd_coarse:
                break
            elif ratio_kd < 1.:
                found_kd_coarse = True
                i_kd -= 99
            elif found_kd_coarse:
                i_kd += 1
            else:
                i_kd += 100
        n_d_kd = n_d_grid[i_kd]
        T_d_kd = T_d_grid[i_kd]
        xi_d_kd = xi_d_grid[i_kd]
        T_kd_3 = Ttrel.T_SM_grid[i_ic+i_kd]
        T_d_kd_3 = T_d_kd
        r_sound_3 = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_grid[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_3_all = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_all[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_3_no_pan = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_no_pan[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # print(T_kd_3, T_d_kd_3, r_sound_3/cf.Mpc, r_sound_3_all/cf.Mpc, r_sound_3_no_pan/cf.Mpc)

        if m_d/T_d_kd > 1e2:
            # only valid if already in non-/semi-relativistic regime
            lo_v_kd = (dof_d*exp(xi_d_kd - m_d/T_d_kd)*(T_d_kd**2.)/(m_d*(np.pi**2.)*n_d_kd)) * \
             ((m_d**2.) + 3.*m_d*T_d_kd + 3.*(T_d_kd**2.))
            
            nlo_v_kd = -(dof_d*exp(xi_d_kd - m_d/T_d_kd)*(T_d_kd**3.)*2./((m_d**3.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**3.) + 6.*(m_d**2.)*T_d_kd + 15.*m_d*(T_d_kd**2.) + 15.*(T_d_kd**3.))
            
            nnlo_v_kd = (dof_d*exp(xi_d_kd - m_d/T_d_kd)*(T_d_kd**4.)*9./((m_d**5.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**4.) + 10.*(m_d**3.)*T_d_kd + 45.*(m_d**2.)*(T_d_kd**2.) + 105.*m_d*(T_d_kd**3.) + 105.*(T_d_kd**4.))

            sf_kd_norm_today = (cf.s0/ent_grid[pan.i_ic+i_kd])**(1./3.)
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid[pan.i_ic+i_kd:i_fs_max])**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            integrand_fs_length = (lo_v_kd/(sf_norm_kd**2.) + nlo_v_kd/(sf_norm_kd**4.) + nnlo_v_kd/(sf_norm_kd**6.))/sf_kd_norm_today
            fs_length_3 = utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:i_fs_max], integrand_fs_length)
        else:
            # always valid
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            p_star_grid = np.logspace(np.log10(1e-6*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), np.log10(1e3*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), 5000)
            f_d_grid = 1./(np.exp(np.sqrt(m_d*m_d+p_star_grid*p_star_grid)/T_d_kd - xi_d_kd) + k_d)
            integrand_fs_length = np.zeros(Ttrel.t_grid.size)
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid)**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            sf_norm_today = (cf.s0/ent_grid)**(1./3.)
            for i in range(i_kd + i_ic, i_fs_max):
                p_cur = p_star_grid/sf_norm_kd[i]
                E_cur = np.sqrt(m_d*m_d + p_cur*p_cur)
                v = utils.simp(p_star_grid, f_d_grid*(p_star_grid**3.)/E_cur)*2./(2.*cf.pi2*n_d_kd*sf_norm_kd[i])
                integrand_fs_length[i] = v/sf_norm_today[i]
            fs_length_3 = utils.simp(Ttrel.t_grid, integrand_fs_length)

        i_kd = i_kd - 1
        i_kd_start = i_kd
        try:
            C_therm_kd_last = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_phi_grid[i_kd])
        except:
            C_therm_kd_last = 0.
        if C_therm_kd_last <= 0.:
            C_therm_kd_last = C_therm_kd_cur * ((T_d_grid[i_kd]/T_d_last)**3.5)
        T_d_last = T_d_grid[i_kd]
        while i_kd < Ttrel.t_grid.size - i_ic - 1:
            try:
                C_therm_kd_cur = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_phi_grid[i_kd])
            except:
                C_therm_kd_cur = 0.
            if C_therm_kd_cur <= 0.:
                C_therm_kd_cur = C_therm_kd_last * ((T_d_grid[i_kd]/T_d_last)**3.5)
            C_therm_kd_last = C_therm_kd_cur
            T_d_last = T_d_grid[i_kd]
            ratio_kd = C_therm_kd_cur / (Ttrel.hubble_grid[i_ic+i_kd]*n_d_grid[i_kd])
            # print('kd', m_d/T_d_grid[i_kd], m_d/T_d_grid[i_kd] - xi_d_grid[i_kd], ratio_kd)
            if ratio_kd < 1.:
                break
            i_kd += 1
        n_d_kd = n_d_grid[i_kd]
        T_d_kd = T_d_grid[i_kd]
        xi_d_kd = xi_d_grid[i_kd]
        T_kd = Ttrel.T_SM_grid[i_ic+i_kd]
        T_d_kd = T_d_kd
        r_sound = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_grid[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_all = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_all[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_no_pan = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_no_pan[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # print(T_kd, T_d_kd, r_sound/cf.Mpc, r_sound_all/cf.Mpc, r_sound_no_pan/cf.Mpc)

        if m_d/T_d_kd > 1e2:
            # only valid if already in non-/semi-relativistic regime
            lo_v_kd = (dof_d*exp(xi_d_kd - m_d/T_d_kd)*(T_d_kd**2.)/(m_d*(np.pi**2.)*n_d_kd)) * \
             ((m_d**2.) + 3.*m_d*T_d_kd + 3.*(T_d_kd**2.))
            
            nlo_v_kd = -(dof_d*exp(xi_d_kd - m_d/T_d_kd)*(T_d_kd**3.)*2./((m_d**3.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**3.) + 6.*(m_d**2.)*T_d_kd + 15.*m_d*(T_d_kd**2.) + 15.*(T_d_kd**3.))
            
            nnlo_v_kd = (dof_d*exp(xi_d_kd - m_d/T_d_kd)*(T_d_kd**4.)*9./((m_d**5.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**4.) + 10.*(m_d**3.)*T_d_kd + 45.*(m_d**2.)*(T_d_kd**2.) + 105.*m_d*(T_d_kd**3.) + 105.*(T_d_kd**4.))

            sf_kd_norm_today = (cf.s0/ent_grid[pan.i_ic+i_kd])**(1./3.)
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid[pan.i_ic+i_kd:i_fs_max])**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            integrand_fs_length = (lo_v_kd/(sf_norm_kd**2.) + nlo_v_kd/(sf_norm_kd**4.) + nnlo_v_kd/(sf_norm_kd**6.))/sf_kd_norm_today
            fs_length = utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:i_fs_max], integrand_fs_length)
        else:
            # always valid
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            p_star_grid = np.logspace(np.log10(1e-6*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), np.log10(1e3*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), 5000)
            f_d_grid = 1./(np.exp(np.sqrt(m_d*m_d+p_star_grid*p_star_grid)/T_d_kd - xi_d_kd) + k_d)
            integrand_fs_length = np.zeros(Ttrel.t_grid.size)
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid)**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            sf_norm_today = (cf.s0/ent_grid)**(1./3.)
            for i in range(i_kd+i_ic, i_fs_max):
                p_cur = p_star_grid/sf_norm_kd[i]
                E_cur = np.sqrt(m_d*m_d + p_cur*p_cur)
                v = utils.simp(p_star_grid, f_d_grid*(p_star_grid**3.)/E_cur)*2./(2.*cf.pi2*n_d_kd*sf_norm_kd[i])
                integrand_fs_length[i] = v/sf_norm_today[i]
            fs_length = utils.simp(Ttrel.t_grid, integrand_fs_length)

        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol, pan.n_chi_grid_sol, pan.n_phi_grid_sol, C_therm_grid, fs_length/cf.Mpc, fs_length_3/cf.Mpc, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound/cf.Mpc, r_sound_3/cf.Mpc, True#, V_SM_grid, V_d_grid, G_a_grid, G_d_grid
    except:
        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol, pan.n_chi_grid_sol, pan.n_phi_grid_sol, C_therm_grid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True

    # print(m_d/T_d_kd, lo_v_kd + nlo_v_kd +nnlo_v_kd)
    # fs_length_cum_int = utils.cumsimp(Ttrel.t_grid[pan.i_ic+i_kd:i_fs_max], integrand_fs_length[:i_fs_max])
    # print(fs_length_cum_int[-1])
    # utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:], sf_kd_norm_today*nlo_v_kd/(sf_norm_kd**4.)), utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:], sf_kd_norm_today*nnlo_v_kd/(sf_norm_kd**6.)))

    # np.savetxt('tmp.dat', np.column_stack((Ttrel.t_grid[i_ic:], Ttrel.T_SM_grid[i_ic:], Ttrel.T_nu_grid[i_ic:], Ttrel.sf_grid[i_ic:], ent_grid[i_ic:], T_d_grid, xi_d_grid, n_d_grid)))
    # np.savetxt('tmp_2.dat', np.array([i_kd, lo_v_kd, nlo_v_kd, nnlo_v_kd]))

    # plt.loglog(Ttrel.T_SM_grid[i_ic+i_kd:]/cf.T0, fs_length_cum_int/cf.Mpc)
    # plt.show()
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:], T_d_grid*(Ttrel.sf_grid[i_ic:]**2.))
    # plt.show()
    # plt.semilogx(m_d/Ttrel.T_nu_grid[i_ic:], xi_d_grid - m_d/T_d_grid)
    # plt.show()
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:], 3.*Ttrel.hubble_grid[i_ic:]*n_d_grid, color='dodgerblue')
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:], C_therm_grid, color='#458751')
    # plt.show()
    # def V_SM(T_SM, T_nu):
    #     rho_nu = 2.*(7./8.)*(cf.pi2/30.)*(T_nu**4.)
    #     rho_a = cf.rho_fermion(T_SM, 0.511e-3, 2.)
    #     return -(8.*sqrt(2)*GF*T_nu/3.)*(rho_nu/(mZ**2.) + rho_a/(mW**2.))
    #
    # def V_d(T_d):
    #     return y2*T_d*T_d/(16.*np.sqrt(T_d*T_d+m_d*m_d)) if T_d > m_phi else -y2*np.sqrt(T_d*T_d+m_d*m_d)*0.75*2.*m_d*m_d*T_d*T_d*np.exp(-m_d/T_d)/(cf.pi2*m_phi2*m_phi2)#-7.*cf.pi2*y2*np.sqrt(T_d*T_d+m_d*m_d)*(T_d**4.)/(90.*m_phi2*m_phi2)
    #
    # def G_a(T_nu):
    #     return 1.27*(GF**2.)*(T_nu**5.)
    #
    # V_SM_grid = np.array([V_SM(T_SM, T_nu) for T_SM, T_nu in zip(Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1])])
    # V_d_grid = np.array([V_d(T_d) for T_d in pan.T_chi_grid_sol])
    # G_a_grid = np.array([G_a(T_nu) for T_nu in Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1]])
    # G_d_grid = np.array([G_d(T_d, xi_d, xi_phi) for T_d, xi_d, xi_phi in zip(pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol)])

if __name__ == '__main__':
    m_d = 2.820886956356105858e-06
    m_phi = 5.*m_d
    m_a = 0.
    k_d = 1.
    k_a = 1.
    k_phi = -1.
    dof_d = 2.
    dof_phi = 1.
    y = 3.736391058821046055e-03
    sin2_2th = 1e-16
    spin_facs = True
    off_shell = False

    t, T_SM, T_nu, ent, H, sf, T_d, xi_d, xi_phi, n_d, n_phi, C_therm, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3, reached_integration_end = call(m_d, m_phi, m_a, k_d, k_phi, k_a, dof_d, dof_phi, sin2_2th, y, spin_facs = spin_facs, off_shell = off_shell)
    print(fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3)

    # filename = f'md_{m_d:.2e}_mphi_{m_phi:.2e}_sin22th_{sin2_2th:.2e}_y_{y:.2e}.dat'
    # np.savetxt('sterile_test/'+filename, np.column_stack((t, T_SM, T_nu, ent, H, sf, T_d, xi_d, xi_phi, n_d, n_phi)))

    import matplotlib.pyplot as plt
    import densities as dens
    rho_d = np.array([dens.rho(k_d, T, m_d, dof_d, xi) for T, xi in zip(T_d, xi_d)])
    rho_phi = np.array([dens.rho(k_phi, T, m_phi, dof_phi, xi) for T, xi in zip(T_d, xi_phi)])
    plt.loglog(m_d/T_nu, m_d/T_d)
    plt.show()
    plt.loglog(m_d/T_nu, n_d*m_d*cf.s0/(ent*cf.rho_crit0_h2), color='dodgerblue')
    plt.loglog(m_d/T_nu, n_phi*m_d*cf.s0/(ent*cf.rho_crit0_h2), color='darkorange')
    plt.loglog(m_d/T_nu, (n_d+n_phi)*m_d*cf.s0/(ent*cf.rho_crit0_h2), color='mediumorchid')
    plt.show()
    plt.close()
    plt.clf()
    plt.loglog(m_d/T_nu, rho_d*(sf**4.), color='dodgerblue')
    plt.loglog(m_d/T_nu, rho_phi*(sf**4.), color='darkorange')
    plt.loglog(m_d/T_nu, (rho_d+rho_phi)*(sf**4.), color='mediumorchid')
    plt.show()
    plt.clf()
    plt.close()
    plt.loglog(m_d/T_nu, C_therm, color='#458751')
    # plt.loglog(m_d/T_nu, C_therm[:,1], color='#458751', ls='--')
    plt.loglog(m_d/T_nu, 3.*H*n_d, color='dodgerblue')
    plt.loglog(m_d/T_nu, 3.*H*n_phi, color='darkorange')
    plt.loglog(m_d/T_nu, 3.*H*np.array([dens.n(k_a, T, m_a, 2., 0.) for T in T_nu]), color='tomato')
    plt.show()
    plt.close()
    plt.clf()
