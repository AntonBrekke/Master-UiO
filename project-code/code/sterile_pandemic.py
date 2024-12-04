#! /usr/bin/env python3

import numpy as np
import numba as nb
from math import cos, sin, asin, sqrt, exp

import constants_functions as cf

import vector_mediator
# import resonant_pandemolator as pandemolator
import pandemolator as pandemolator
import C_res_vector

m_d = 1e-5
m_a = 0.
m_X = 2.5*m_d
sin2_2th = 1e-12
y = 5.6e-5

k_d = 1.
k_a = 1.
k_phi = -1.
dof_d = 2.
dof_phi = 1.

m_d2 = m_d*m_d
m_a2 = m_a*m_a
m_X2 = m_X*m_X
th = 0.5*asin(sqrt(sin2_2th))
c_th = cos(th)
s_th = sin(th)
y2 = y*y

# Anton: Matrix elements added here for some reason
# M2_dd = 2. * y2 * (c_th**4.) * (m_X2 - 4.*m_d2)
# M2_aa = 2. * y2 * (s_th**4.) * (m_X2 - 4.*m_a2)
# M2_da = 2. * y2 * (s_th**2.) * (c_th**2.) * (m_X2 - ((m_a+m_d)**2.))

# M2_X23 = 2*g**2/m_X2 * (m_X2 - (m2 - m3)**2)*(2*m_X2 + (m2 + m3)**2)
# New matrix elements for X --> 23
M2_dd = 2.*y2*(c_th**4.)/m_X2 * (m_X2)*(2*m_X2 + (m_d + m_d)**2)
M2_aa = 2.*y2*(s_th**4.)/m_X2 * (m_X2)*(2*m_X2)
M2_da = 2.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d**2)*(2*m_X2 + m_d**2)

vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
vert_el = y2*y2*(c_th**8.)

Gamma_X = vector_mediator.Gamma_X(y, th, m_X, m_d)
m_Gamma_X2 = m_X2*Gamma_X*Gamma_X

# n = n_d + 2.*n_phi
def C_n(T_a, T_d, xi_d, xi_phi):
    C_XX_dd = C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_phi, T_d, xi_d, xi_phi, vert_el) / 4. # symmetry factor 1/4
    C_da = C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d,   0., xi_phi, M2_da)
    C_aa = C_res_vector.C_n_3_12(m_a, m_a, m_X, k_a, k_a, k_phi, T_a, T_a, T_d,   0.,   0., xi_phi, M2_aa) / 2.
    C_da_dd = C_res_vector.C_34_12(0, 1., -1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2) / 2.
    C_aa_da = C_res_vector.C_34_12(0, 1., -1., m_d, m_a, m_a, m_a, k_d, k_a, k_a, k_a, T_d, T_a, T_a, T_a, xi_d, 0., 0., 0., vert_fi, m_X2, m_Gamma_X2) / 2.
    print("C_ns:", C_da, 2.*C_aa, C_da_dd, C_aa_da, 2.*C_XX_dd)
    return C_da + 2.*C_aa + C_da_dd + C_aa_da + 2.*C_XX_dd

# rho = rho_d + rho_phi
def C_rho(T_a, T_d, xi_d, xi_phi):
    C_da = C_res_vector.C_rho_3_12(2, m_d, m_a, m_X, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d, 0., xi_phi, M2_da)
    C_aa = C_res_vector.C_rho_3_12(3, m_a, m_a, m_X, k_d, k_a, k_phi, T_a, T_a, T_d,   0., 0., xi_phi, M2_aa) / 2. # symmetry factor 1/2
    C_da_dd = C_res_vector.C_34_12(4, 1., -1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2) / 2.
    C_aa_da = C_res_vector.C_34_12(1, 1., -1., m_d, m_a, m_a, m_a, k_d, k_a, k_a, k_a, T_d, T_a, T_a, T_a, xi_d, 0., 0., 0., vert_fi, m_X2, m_Gamma_X2) / 2.
    print("C_rhos:", C_da, C_aa, C_da_dd, C_aa_da)
    return C_da + C_aa + C_da_dd + C_aa_da

def C_xi0(T_a, T_d, xi_d, xi_phi):
    C_XX_dd = C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_phi, T_d, xi_d, xi_phi, vert_el, type=1) / 4.
    return 2.*C_XX_dd

Ttrel = pandemolator.TimeTempRelation()
ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
T_d_DW = 0.133*((1e6*m_d)**1./3.) # temperature of maximal d production by Dodelson-Widrow mechanism
i_ic = np.argmax(Ttrel.T_nu_grid < T_d_DW)
i_end = np.argmax(Ttrel.T_nu_grid < 0.01*m_d)
sf_ic_norm_d_DW = (cf.s_SM_before_nu_dec(T_d_DW)/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
O_d_h2 = 0.3*1e10*sin2_2th*((1e4*m_d)**2.)
norm_f_d_0 = 4.*cf.pi2*cf.s_SM_before_nu_dec(T_d_DW)*O_d_h2*cf.rho_crit0_h2/(3.*cf.zeta3*(T_d_DW**3.)*m_d*cf.s0)
T_d_ic = ((norm_f_d_0/(1.+8.*dof_phi/(7.*dof_d)))**(1./3.))*T_d_DW/sf_ic_norm_d_DW
xi_d_ic = 0.
xi_phi_ic = 0.

sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
n_ic = cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)
rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0

# pan = pandemolator.Pandemolator(m_d, k_d, dof_d, m_X, k_phi, dof_phi, m_a, k_a, C_n, C_rho, C_xi0, Ttrel.t_grid, Ttrel.T_nu_grid, Ttrel.dTnu_dt_grid, ent_grid, Ttrel.hubble_grid, Ttrel.sf_grid, i_ic, T_d_ic, xi_d_ic, xi_phi_ic, i_end)
pan = pandemolator.Pandemolator(m_d, k_d, dof_d, m_X, k_phi, dof_phi, m_a, k_a, C_n, C_rho, C_xi0, Ttrel.t_grid, Ttrel.T_nu_grid, Ttrel.dTnu_dt_grid, ent_grid, Ttrel.hubble_grid, Ttrel.sf_grid, i_ic, n_ic, rho_ic, i_end)
pan.pandemolate()

np.savetxt('sterile_test/md_1e-5_mX_2.5e-5_sin22th_1e-12_y_5.6e-5_full.dat', np.column_stack((Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_phi_grid_sol, pan.n_chi_grid_sol, pan.n_phi_grid_sol)))
