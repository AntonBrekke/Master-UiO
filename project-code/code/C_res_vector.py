#! /usr/bin/env python3

from scipy.integrate import quad
import numpy as np
import numba as nb
import cmath as cm
from math import exp, log, sqrt, pi, fabs, atan, asin, tan, isfinite
import vegas
from scipy.special import kn
import densities as dens
import scalar_mediator

max_exp_arg = 3e2
rtol_int = 1e-4
spin_stat_irr = 1e3
fac_res_width = 1e4
offset = 1.+1e-14

"""
Changes: 
Updated er_C_n_pp_dd_s_t_integral to ker_C_n_XX_dd_s_t_integral (_2)
Updated sigma_pp_dd to sigma_XX_dd

Have to do: 

Change 3 --> 12:
ker_C_n_3_12_E2 
ker_C_n_3_12_E1
Gamma_scat
C_n_3_12

Change 34 --> 12:
ker_C_34_12_s_t_integral
ker_th_avg_sigma_v_33_11 
"""

@nb.jit(nopython=True, cache=True)
def ker_C_n_3_12_E2(log_E2, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type):
    E2 = exp(log_E2)
    E3 = E1 + E2
    exp_arg_2 = E2/T2 - xi2
    exp_arg_3 = E3/T3 - xi3
    f2 = 1./(exp(exp_arg_2) + k2) if exp_arg_2 < max_exp_arg else 0.
    f3 = 1./(exp(exp_arg_3) + k3) if exp_arg_3 < max_exp_arg else 0.

    if type == 0:
        if T1 == T2 and T1 == T3:
            chem_eq_fac = 1.-exp(-xi1-xi2+xi3)
            dist = f1*f2*(1.-k3*f3)*chem_eq_fac
        else:
            dist_1 = f1*f2*(1-k3*f3)
            dist_2 = f3*(1-k1*f1)*(1-k2*f2)
            dist = dist_1 - dist_2
            if fabs(dist) <= 1e-12*max(fabs(dist_1), fabs(dist_2)):
                return 0.
    elif type == -1:
        dist = -f3*(1.-k1*f1)*(1.-k2*f2)
    elif type == 1:
        dist = f1*f2*(1.-k3*f3)

    res = E2*dist
    if not isfinite(res):
        return 0.
    return res

def ker_C_n_3_12_E1(log_E1, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type):
    E1 = exp(log_E1)
    p1 = sqrt((E1 - m1)*(E1 + m1)) if E1 > m1 else 0.

    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    if m1 > 0.:
        sqrt_arg = m12*m12 + (((m2-m3)*(m2+m3))**2.) - 2.*m12*(m22+m32)
        if sqrt_arg <= 0.:
            return 0.
        sqrt_fac = sqrt(sqrt_arg)
        E2_min = max(m2, (E1*(m32-m12-m22) - p1*sqrt_fac)/(2.*m12), 1e-200)
        E2_max = (E1*(m32-m12-m22) + p1*sqrt_fac)/(2.*m12)
    else:
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1), 1e-200)
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)
    if E2_max <= E2_min:
        return 0.

    exp_arg_1 = E1/T1 - xi1
    f1 = 1./(exp(exp_arg_1) + k1) if exp_arg_1 < max_exp_arg else 0.

    res_2, err = quad(ker_C_n_3_12_E2, log(E2_min), log(E2_max), args=(E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type), epsabs=0., epsrel=rtol_int)

    res = E1*res_2
    if not isfinite(res):
        return 0.
    return res

def Gamma_scat(p1, m1, m2, m3, k2, k3, T2, T3, xi2, xi3, M2):
    E1 = sqrt(p1*p1+m1*m1)
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    if m1 > 0.:
        sqrt_arg = m12*m12 + (((m2-m3)*(m2+m3))**2.) - 2.*m12*(m22+m32)
        if sqrt_arg <= 0.:
            return 0.
        sqrt_fac = sqrt(sqrt_arg)
        E2_min = max(m2, (E1*(m32-m12-m22) - p1*sqrt_fac)/(2.*m12), 1e-200)
        E2_max = (E1*(m32-m12-m22) + p1*sqrt_fac)/(2.*m12)
    else:
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1), 1e-200)
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)
    if E2_max <= E2_min:
        return 0.
    res, err = quad(ker_C_n_3_12_E2, log(E2_min), log(E2_max), args=(E1, 1., 0., k2, k3, E1, T2, T3, 0., xi2, xi3, 1), epsabs=0., epsrel=rtol_int)
    return M2*res/(16.*pi*p1*E1)

# type == -1: only 3 -> 1 2, type == 0: both reactions, type == 1: only 1 2 -> 3
def C_n_3_12(m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, M2, type = 0):
    E1_min = max(m1, 1e-200)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)

    res, err = quad(ker_C_n_3_12_E1, log(E1_min), log(E1_max), args=(m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type), epsabs=0., epsrel=rtol_int)

    return M2*res/(32.*(pi**3.))

# E_type * (f1*f2*(1+f3) - f3*(1-f1)*(1-f2))
@nb.jit(nopython=True, cache=True)
def ker_C_rho_3_12_E2(log_E2, type, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3):
    E2 = exp(log_E2)
    E3 = E1 + E2
    exp_arg_2 = E2/T2 - xi2
    exp_arg_3 = E3/T3 - xi3
    f2 = 1./(exp(exp_arg_2) + k2) if exp_arg_2 < max_exp_arg else 0.
    f3 = 1./(exp(exp_arg_3) + k3) if exp_arg_3 < max_exp_arg else 0.

    dist_1 = f1*f2*(1-k3*f3)
    dist_2 = f3*(1-k1*f1)*(1-k2*f2)
    dist = dist_1 - dist_2
    if fabs(dist) <= 1e-12*max(fabs(dist_1), fabs(dist_2)):
        return 0.

    if type == 1.:
        Etype = E1
    elif type == 2.:
        Etype = E2
    else:
        Etype = E3

    res = E2*Etype*dist
    if not isfinite(res):
        return 0.
    return res

def ker_C_rho_3_12_E1(log_E1, type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3):
    E1 = exp(log_E1)
    p1 = sqrt((E1 - m1)*(E1 + m1)) if E1 > m1 else 0.

    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    if m1 > 0.:
        sqrt_arg = m12*m12 + (((m2-m3)*(m2+m3))**2.) - 2.*m12*(m22+m32)
        if sqrt_arg <= 0.:
            return 0.
        sqrt_fac = sqrt(sqrt_arg)
        E2_min = max(m2, (E1*(m32-m12-m22) - p1*sqrt_fac)/(2.*m12), 1e-200)
        E2_max = (E1*(m32-m12-m22) + p1*sqrt_fac)/(2.*m12)
    else:
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1), 1e-200)
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)
        # print(E1/T1, E2_min/T2, E2_max/T2, E1*m22/((m3-m2)*(m3+m2))/T2, ((m3-m2)*(m3+m2))/(4.*E1)/T2)
    if E2_max <= E2_min:
        return 0.

    exp_arg_1 = E1/T1 - xi1
    f1 = 1./(exp(exp_arg_1) + k1) if exp_arg_1 < max_exp_arg else 0.

    res_2, err = quad(ker_C_rho_3_12_E2, log(E2_min), log(E2_max), args=(type, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3), epsabs=0., epsrel=rtol_int)

    res = E1*res_2
    if not isfinite(res):
        return 0.
    return res

def C_rho_3_12(type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, M2):
    E1_min = max(m1, 1e-10*T1)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)

    res, err = quad(ker_C_rho_3_12_E1, log(E1_min), log(E1_max), args=(type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3), epsabs=0., epsrel=rtol_int)

    return M2*res/(32.*(pi**3.))

# This kernel is replaced by 'ker_C_n_XX_dd_s_t_integral_2'. 
@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s_t_integral(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_X, vert):
    """
    Solved for matrix-element and integrated over variable t in Mathematica. 
    These expressions can be found in nu_s_nu_s_to_XX.nb
    """
    s2 = s*s
    m_X2 = m_X*m_X
    m_d2 = m_d*m_d
    m_X4 = m_X2*m_X2
    m_d4 = m_d2*m_d2
    m_X6 = m_X4*m_X2
    m_d6 = m_d4*m_d2
    m_X8 = m_X4*m_X4
    m_d8 = m_d4*m_d4

    t_add = m_d2 + m_X2
    t_min = t_add - 2.*E1*E3 + 2*p1*p3*ct_min
    t_max = t_add - 2.*E1*E3 + 2*p1*p3*ct_max
    t_m = t_add - 2.*E1*E3 + 2*p1*p3*ct_m
    t_p = t_add - 2.*E1*E3 + 2*p1*p3*ct_p

    in_min_neq = (ct_min != ct_p)
    in_max_neq = (ct_max != ct_m)
    n = s.size

    # Write each term of t-integrated matrix element sorted by denominators, evaluated at t_min and t_max. 
    """
    sqrt(x)*sqrt(y) / sqrt(x*y) makes trouble. 
    This is either 1 for (x>=0, y>=0), (x>=0, y<0), (x<0, y>=0) and -1 for (x<0, y<0).
    Also, t_p > t_m always, so write log(t_m - t_p) = i*pi + log(t_p - t_m)
    """
    sqrt_fac_tmin = 1*(t_min-t_m >= 0)*(t_min-t_p >= 0) + 1*(t_min-t_m >= 0)*(t_min-t_p < 0) + 1*(t_min-t_m < 0)*(t_min-t_p >= 0) - 1*(t_min-t_m < 0)*(t_min-t_p < 0)
    sqrt_fac_tmax = 1*(t_max-t_m >= 0)*(t_max-t_p >= 0) + 1*(t_max-t_m >= 0)*(t_max-t_p < 0) + 1*(t_max-t_m < 0)*(t_max-t_p >= 0) - 1*(t_max-t_m < 0)*(t_max-t_p < 0)

    # Trick to make numpy evaluate negative numbers 
    t_min = t_min + 0j
    t_max = t_max + 0j
    t_m = t_m + 0j
    t_p = t_p + 0j
    s = s + 0j
    s2 = s2 + 0j

    # Write each term evaluated manually in limits t_min = t_p and t_max = t_m
    # term1: vanish due to sqrt((t - t_p)*(t - t_m))
    term1_min = np.zeros(n) + 0j
    term1_max = np.zeros(n) + 0j

    # term2: -----
    term2_min = -((4*sqrt_fac_tmin*(16*m_d8-16*m_d6*(t_m+t_p) - 2*m_d4*(12*m_X4+4*m_X2*(-s+t_m+t_p) + s**2-2*s*(t_m+t_p)-8*t_m*t_p) + 2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p)) - 2*m_X2*(s*(t_m+t_p) - 4*t_m*t_p) + s*(s*(t_m+t_p)-4*t_m*t_p)) + 2*m_X6*(t_m+t_p) - m_X4*(s*(t_m+t_p)+8*t_m*t_p) - 2*s**2*t_m*t_p)*(np.log(m_d2-t_p) - np.pi*1j - np.log(m_d2*(t_p-t_m) - t_p*(t_m+t_p) + 2*t_m*t_p))) / ((m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)*(2*m_X2-s)))

    term2_max = -((4*sqrt_fac_tmax*(16*m_d8-16*m_d6*(t_m+t_p) - 2*m_d4*(12*m_X4+4*m_X2*(-s+t_m+t_p) + s**2-2*s*(t_m+t_p)-8*t_m*t_p) + 2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p)) - 2*m_X2*(s*(t_m+t_p) - 4*t_m*t_p) + s*(s*(t_m+t_p)-4*t_m*t_p)) + 2*m_X6*(t_m+t_p) - m_X4*(s*(t_m+t_p)+8*t_m*t_p) - 2*s**2*t_m*t_p)*(np.log(m_d2-t_m) - np.log(m_d2*(t_m-t_p) - t_m*(t_m+t_p) + 2*t_m*t_p))) / ((m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)*(2*m_X2-s)))

    # term3: -----
    term3_min = (4*sqrt_fac_tmin*(16*m_d8+16*m_d6*(6*m_X2-3*s-t_m-t_p) + 2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p) + 8*t_m*t_p)+2*m_d2*(34*m_X6-m_X4*(57*s+16*(t_m+t_p)) + 2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)-s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p)) - 24*m_X8+2*m_X6*(12*s+7*(t_m+t_p))-m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)+4*m_X2*s2*(2*s+t_m+t_p)-2*s2*(s+t_m)*(s+t_p))*(np.log(m_d2+2*m_X2-s-t_p) - np.log(m_d2*(2*t_p-t_m-t_p)+m_X2*(4*t_p-2*(t_m+t_p)) + t_p*(s-t_p+2*t_m)-2*s*t_p+s*t_m-t_p*t_m))) / ((2*m_X2-s)*(m_d2+2*m_X2-s-t_m)**(3/2)*(m_d2+2*m_X2-s-t_p)**(3/2)) 

    term3_max = (4*sqrt_fac_tmax*(16*m_d8+16*m_d6*(6*m_X2-3*s-t_m-t_p) + 2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p) + 8*t_m*t_p)+2*m_d2*(34*m_X6-m_X4*(57*s+16*(t_m+t_p)) + 2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)-s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p)) - 24*m_X8+2*m_X6*(12*s+7*(t_m+t_p))-m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)+4*m_X2*s2*(2*s+t_m+t_p)-2*s2*(s+t_m)*(s+t_p))*(np.log(m_d2+2*m_X2-s-t_m) - np.log(m_d2*(2*t_m-t_m-t_p)+m_X2*(4*t_m-2*(t_m+t_p)) + t_p*(s-t_m+2*t_m)-2*s*t_m+s*t_m-t_m*t_m))) / ((2*m_X2-s)*(m_d2+2*m_X2-s-t_m)**(3/2)*(m_d2+2*m_X2-s-t_p)**(3/2))

    # term4: -----
    term4_min = -(16*sqrt_fac_tmin*np.log(t_p-t_m))
    term4_max = -(16*sqrt_fac_tmax*np.log(t_m-t_p))     

    # Write each term of t-integrated matrix element sorted by denominators, evaluated at t_min and t_max. 
    # These are for t_min neq t_p and t_max neq t_m, so need full expressions
    if np.any(in_min_neq):
        in_t_min_neq = np.logical_and(in_min_neq, t_min!=t_p)
        t_min_in_neq = t_min[in_t_min_neq]
        t_m_in_neq = t_m[in_t_min_neq]
        t_p_in_neq = t_p[in_t_min_neq]
        s_in_neq = s[in_t_min_neq]
        s2_in_neq = s2[in_t_min_neq]

        term1_min[in_t_min_neq] = -((8*(2*m_d2+m_X2)**2*np.sqrt((t_min_in_neq-t_m_in_neq)*(t_min_in_neq-t_p_in_neq))*(2*m_d6+m_d4*(6*m_X2-3*s_in_neq-2*(t_min_in_neq+t_m_in_neq+t_p_in_neq)) + m_d2*(12*m_X4-4*m_X2*(3*s_in_neq+t_min_in_neq+t_m_in_neq+t_p_in_neq) + 3*s2_in_neq+2*s_in_neq*(t_min_in_neq+t_m_in_neq+t_p_in_neq) + 2*t_p_in_neq*(t_min_in_neq+t_m_in_neq)+2*t_min_in_neq*t_m_in_neq) + (2*m_X2-s_in_neq)*(2*m_X2-s_in_neq-t_min_in_neq)*(2*m_X2-s_in_neq-t_m_in_neq) - t_p_in_neq*(4*m_X4-2*m_X2*(2*s_in_neq+t_min_in_neq+t_m_in_neq) + s2_in_neq+s_in_neq*(t_min_in_neq+t_m_in_neq)+2*t_min_in_neq*t_m_in_neq))) / ((t_min_in_neq-m_d2)*(t_m_in_neq-m_d2)*(t_p_in_neq-m_d2)*(-m_d2-2*m_X2+s_in_neq+t_min_in_neq)*(-m_d2-2*m_X2+s_in_neq+t_m_in_neq)*(-m_d2-2*m_X2+s_in_neq+t_p_in_neq)))

        term2_min[in_t_min_neq] = -((4*np.sqrt((t_min_in_neq-t_m_in_neq)*(t_min_in_neq-t_p_in_neq))*(16*m_d8-16*m_d6*(t_m_in_neq+t_p_in_neq) - 2*m_d4*(12*m_X4+4*m_X2*(-s_in_neq+t_m_in_neq+t_p_in_neq) + s_in_neq**2-2*s_in_neq*(t_m_in_neq+t_p_in_neq)-8*t_m_in_neq*t_p_in_neq) + 2*m_d2*(-2*m_X6+m_X4*(s_in_neq+8*(t_m_in_neq+t_p_in_neq)) - 2*m_X2*(s_in_neq*(t_m_in_neq+t_p_in_neq) - 4*t_m_in_neq*t_p_in_neq) + s_in_neq*(s_in_neq*(t_m_in_neq+t_p_in_neq)-4*t_m_in_neq*t_p_in_neq)) + 2*m_X6*(t_m_in_neq+t_p_in_neq) - m_X4*(s_in_neq*(t_m_in_neq+t_p_in_neq)+8*t_m_in_neq*t_p_in_neq) - 2*s_in_neq**2*t_m_in_neq*t_p_in_neq)*(np.log(m_d2-t_min_in_neq) - np.log(m_d2*(2*t_min_in_neq-t_m_in_neq-t_p_in_neq) + 2*np.sqrt(m_d2-t_m_in_neq)*np.sqrt(m_d2-t_p_in_neq)*np.sqrt(t_min_in_neq-t_m_in_neq)*np.sqrt(t_min_in_neq-t_p_in_neq) - t_min_in_neq*(t_m_in_neq+t_p_in_neq) + 2*t_m_in_neq*t_p_in_neq))) / ((m_d2-t_m_in_neq)**(3/2)*(m_d2-t_p_in_neq)**(3/2)*(2*m_X2-s_in_neq)*np.sqrt(t_min_in_neq-t_m_in_neq)*np.sqrt(t_min_in_neq-t_p_in_neq)))

        term3_min[in_t_min_neq] = (4*np.sqrt((t_min_in_neq-t_m_in_neq)*(t_min_in_neq-t_p_in_neq))*(16*m_d8+16*m_d6*(6*m_X2-3*s_in_neq-t_m_in_neq-t_p_in_neq) + 2*m_d4*(84*m_X4-28*m_X2*(3*s_in_neq+t_m_in_neq+t_p_in_neq)+19*s2_in_neq+14*s_in_neq*(t_m_in_neq+t_p_in_neq) + 8*t_m_in_neq*t_p_in_neq)+2*m_d2*(34*m_X6-m_X4*(57*s_in_neq+16*(t_m_in_neq+t_p_in_neq)) + 2*m_X2*(12*s2_in_neq+9*s_in_neq*(t_m_in_neq+t_p_in_neq)+4*t_m_in_neq*t_p_in_neq)-s_in_neq*(2*s2_in_neq+3*s_in_neq*(t_m_in_neq+t_p_in_neq)+4*t_m_in_neq*t_p_in_neq)) - 24*m_X8+2*m_X6*(12*s_in_neq+7*(t_m_in_neq+t_p_in_neq))-m_X4*(14*s2_in_neq+7*s_in_neq*(t_m_in_neq+t_p_in_neq)+8*t_m_in_neq*t_p_in_neq)+4*m_X2*s2_in_neq*(2*s_in_neq+t_m_in_neq+t_p_in_neq)-2*s2_in_neq*(s_in_neq+t_m_in_neq)*(s_in_neq+t_p_in_neq))*(np.log(m_d2+2*m_X2-s_in_neq-t_min_in_neq) - np.log(2*np.sqrt(t_min_in_neq-t_m_in_neq)*np.sqrt(t_min_in_neq-t_p_in_neq)*np.sqrt(m_d2+2*m_X2-s_in_neq-t_m_in_neq)*np.sqrt(m_d2+2*m_X2-s_in_neq-t_p_in_neq) + m_d2*(2*t_min_in_neq-t_m_in_neq-t_p_in_neq)+m_X2*(4*t_min_in_neq-2*(t_m_in_neq+t_p_in_neq)) + t_p_in_neq*(s_in_neq-t_min_in_neq+2*t_m_in_neq)-2*s_in_neq*t_min_in_neq+s_in_neq*t_m_in_neq-t_min_in_neq*t_m_in_neq))) / ((2*m_X2-s_in_neq)*np.sqrt(t_min_in_neq-t_m_in_neq)*np.sqrt(t_min_in_neq-t_p_in_neq)*(m_d2+2*m_X2-s_in_neq-t_m_in_neq)**(3/2)*(m_d2+2*m_X2-s_in_neq-t_p_in_neq)**(3/2))

        term4_min[in_t_min_neq] = -((16*np.sqrt((t_min_in_neq-t_m_in_neq)*(t_min_in_neq-t_p_in_neq))*np.log(2*np.sqrt(t_min_in_neq-t_m_in_neq)*np.sqrt(t_min_in_neq-t_p_in_neq)+2*t_min_in_neq-t_m_in_neq-t_p_in_neq)) / (np.sqrt(t_min_in_neq-t_m_in_neq)*np.sqrt(t_min_in_neq-t_p_in_neq)))
        # print('In min_neq')

    if np.any(in_max_neq):
        in_t_max_neq = np.logical_and(in_max_neq, t_max!=t_m)
        t_max_in_neq = t_max[in_t_max_neq]
        t_m_in_neq = t_m[in_t_max_neq]
        t_p_in_neq = t_p[in_t_max_neq]
        s_in_neq = s[in_t_max_neq]
        s2_in_neq = s2[in_t_max_neq]

        term1_max[in_t_max_neq] = -((8*(2*m_d2+m_X2)**2*np.sqrt((t_max_in_neq-t_m_in_neq)*(t_max_in_neq-t_p_in_neq))*(2*m_d6+m_d4*(6*m_X2-3*s_in_neq-2*(t_max_in_neq+t_m_in_neq+t_p_in_neq)) + m_d2*(12*m_X4-4*m_X2*(3*s_in_neq+t_max_in_neq+t_m_in_neq+t_p_in_neq) + 3*s2_in_neq+2*s_in_neq*(t_max_in_neq+t_m_in_neq+t_p_in_neq) + 2*t_p_in_neq*(t_max_in_neq+t_m_in_neq)+2*t_max_in_neq*t_m_in_neq) + (2*m_X2-s_in_neq)*(2*m_X2-s_in_neq-t_max_in_neq)*(2*m_X2-s_in_neq-t_m_in_neq) - t_p_in_neq*(4*m_X4-2*m_X2*(2*s_in_neq+t_max_in_neq+t_m_in_neq) + s2_in_neq+s_in_neq*(t_max_in_neq+t_m_in_neq)+2*t_max_in_neq*t_m_in_neq))) / ((t_max_in_neq-m_d2)*(t_m_in_neq-m_d2)*(t_p_in_neq-m_d2)*(-m_d2-2*m_X2+s_in_neq+t_max_in_neq)*(-m_d2-2*m_X2+s_in_neq+t_m_in_neq)*(-m_d2-2*m_X2+s_in_neq+t_p_in_neq)))

        term2_max[in_t_max_neq] = -((4*np.sqrt((t_max_in_neq-t_m_in_neq)*(t_max_in_neq-t_p_in_neq))*(16*m_d8-16*m_d6*(t_m_in_neq+t_p_in_neq) - 2*m_d4*(12*m_X4+4*m_X2*(-s_in_neq+t_m_in_neq+t_p_in_neq) + s_in_neq**2-2*s_in_neq*(t_m_in_neq+t_p_in_neq)-8*t_m_in_neq*t_p_in_neq) + 2*m_d2*(-2*m_X6+m_X4*(s_in_neq+8*(t_m_in_neq+t_p_in_neq)) - 2*m_X2*(s_in_neq*(t_m_in_neq+t_p_in_neq) - 4*t_m_in_neq*t_p_in_neq) + s_in_neq*(s_in_neq*(t_m_in_neq+t_p_in_neq)-4*t_m_in_neq*t_p_in_neq)) + 2*m_X6*(t_m_in_neq+t_p_in_neq) - m_X4*(s_in_neq*(t_m_in_neq+t_p_in_neq)+8*t_m_in_neq*t_p_in_neq) - 2*s_in_neq**2*t_m_in_neq*t_p_in_neq)*(np.log(m_d2-t_max_in_neq) - np.log(m_d2*(2*t_max_in_neq-t_m_in_neq-t_p_in_neq) + 2*np.sqrt(m_d2-t_m_in_neq)*np.sqrt(m_d2-t_p_in_neq)*np.sqrt(t_max_in_neq-t_m_in_neq)*np.sqrt(t_max_in_neq-t_p_in_neq) - t_max_in_neq*(t_m_in_neq+t_p_in_neq) + 2*t_m_in_neq*t_p_in_neq))) / ((m_d2-t_m_in_neq)**(3/2)*(m_d2-t_p_in_neq)**(3/2)*(2*m_X2-s_in_neq)*np.sqrt(t_max_in_neq-t_m_in_neq)*np.sqrt(t_max_in_neq-t_p_in_neq)))

        term3_max[in_t_max_neq] = (4*np.sqrt((t_max_in_neq-t_m_in_neq)*(t_max_in_neq-t_p_in_neq))*(16*m_d8+16*m_d6*(6*m_X2-3*s_in_neq-t_m_in_neq-t_p_in_neq) + 2*m_d4*(84*m_X4-28*m_X2*(3*s_in_neq+t_m_in_neq+t_p_in_neq)+19*s2_in_neq+14*s_in_neq*(t_m_in_neq+t_p_in_neq) + 8*t_m_in_neq*t_p_in_neq)+2*m_d2*(34*m_X6-m_X4*(57*s_in_neq+16*(t_m_in_neq+t_p_in_neq)) + 2*m_X2*(12*s2_in_neq+9*s_in_neq*(t_m_in_neq+t_p_in_neq)+4*t_m_in_neq*t_p_in_neq)-s_in_neq*(2*s2_in_neq+3*s_in_neq*(t_m_in_neq+t_p_in_neq)+4*t_m_in_neq*t_p_in_neq)) - 24*m_X8+2*m_X6*(12*s_in_neq+7*(t_m_in_neq+t_p_in_neq))-m_X4*(14*s2_in_neq+7*s_in_neq*(t_m_in_neq+t_p_in_neq)+8*t_m_in_neq*t_p_in_neq)+4*m_X2*s2_in_neq*(2*s_in_neq+t_m_in_neq+t_p_in_neq)-2*s2_in_neq*(s_in_neq+t_m_in_neq)*(s_in_neq+t_p_in_neq))*(np.log(m_d2+2*m_X2-s_in_neq-t_max_in_neq) - np.log(2*np.sqrt(t_max_in_neq-t_m_in_neq)*np.sqrt(t_max_in_neq-t_p_in_neq)*np.sqrt(m_d2+2*m_X2-s_in_neq-t_m_in_neq)*np.sqrt(m_d2+2*m_X2-s_in_neq-t_p_in_neq) + m_d2*(2*t_max_in_neq-t_m_in_neq-t_p_in_neq)+m_X2*(4*t_max_in_neq-2*(t_m_in_neq+t_p_in_neq)) + t_p_in_neq*(s_in_neq-t_max_in_neq+2*t_m_in_neq)-2*s_in_neq*t_max_in_neq+s_in_neq*t_m_in_neq-t_max_in_neq*t_m_in_neq))) / ((2*m_X2-s_in_neq)*np.sqrt(t_max_in_neq-t_m_in_neq)*np.sqrt(t_max_in_neq-t_p_in_neq)*(m_d2+2*m_X2-s_in_neq-t_m_in_neq)**(3/2)*(m_d2+2*m_X2-s_in_neq-t_p_in_neq)**(3/2))

        term4_max[in_t_max_neq] = -((16*np.sqrt((t_max_in_neq-t_m_in_neq)*(t_max_in_neq-t_p_in_neq))*np.log(2*np.sqrt(t_max_in_neq-t_m_in_neq)*np.sqrt(t_max_in_neq-t_p_in_neq)+2*t_max_in_neq-t_m_in_neq-t_p_in_neq)) / (np.sqrt(t_max_in_neq-t_m_in_neq)*np.sqrt(t_max_in_neq-t_p_in_neq)))
        # print('In max_neq')

    # print(vert*(term1_max + term2_max + term3_max + term4_max - term1_min - term2_min - term3_min - term4_min).imag)
    # print(vert*(term1_max + term2_max + term3_max + term4_max - term1_min - term2_min - term3_min - term4_min).real)
    print(np.min(term1_min), np.max(term1_max))
    print(np.min(term2_min), np.max(term2_max))
    print(np.min(term3_min), np.max(term3_max))
    print(np.min(term4_min), np.max(term4_max))
    return vert*(term1_max + term2_max + term3_max + term4_max - term1_min - term2_min - term3_min - term4_min).real

# For XX --> dd (for some reason, instead of dd --> XX)
@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s_t_integral_2(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_X, vert):
    """
    Solved for matrix-element and integrated over variable t in Mathematica. 
    These expressions can be found in nu_s_nu_s_to_XX.nb
    """
    n = s.size
    s2 = s*s
    m_d2 = m_d*m_d
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    t_add = m_d2 + m_X2
    t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
    t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
    t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)
    
    # Write each term of t-integrated matrix element sorted by denominators, evaluated at t_min and t_max. 
    """
    sqrt(x)*sqrt(y) / sqrt(x*y) makes trouble. 
    This is either 1 for (x>=0, y>=0), (x>=0, y<0), (x<0, y>=0) and -1 for (x<0, y<0).
    Also, t_p > t_m always, so write log(t_m - t_p) = i*pi + log(t_p - t_m)
    """
    sqrt_fac_tmin = np.zeros(n)
    sqrt_fac_tmax = np.zeros(n)
    X_max = t_max - t_m
    Y_max = t_max - t_p
    X_min = t_min - t_m
    Y_min = t_min - t_p

    sqrt_fac_tmax[np.logical_and(X_max >= 0, Y_max >= 0)] = 1
    sqrt_fac_tmax[np.logical_and(X_max < 0, Y_max >= 0)] = 1
    sqrt_fac_tmax[np.logical_and(X_max >= 0, Y_max < 0)] = 1
    sqrt_fac_tmax[np.logical_and(X_max < 0, Y_max < 0)] = -1
    
    sqrt_fac_tmin[np.logical_and(X_min >= 0, Y_min >= 0)] = 1
    sqrt_fac_tmin[np.logical_and(X_min < 0, Y_min >= 0)] = 1
    sqrt_fac_tmin[np.logical_and(X_min >= 0, Y_min < 0)] = 1
    sqrt_fac_tmin[np.logical_and(X_min < 0, Y_min < 0)] = -1
    
    # Trick to make numpy evaluate negative numbers 
    t_min = t_min + 0j
    t_max = t_max + 0j
    t_m = t_m + 0j
    t_p = t_p + 0j
    s = s + 0j
    s2 = s2 + 0j
    
    # When t_max = t_m, t_min = t_p, max - min always cancel ... 
    term1_max = -((8*(2*m_d2+m_X2)**2*np.sqrt((t_max-t_m)*(t_max-t_p))*(2*m_d6+m_d4*(6*m_X2-3*s-2*(t_max+t_m+t_p))+m_d2*(12*m_X4-4*m_X2*(3*s+t_max+t_m+t_p)+3*s2+2*s*(t_max+t_m+t_p)+2*t_p*(t_max+t_m)+2*t_max*t_m)+(2*m_X2-s)*(2*m_X2-s-t_max)*(2*m_X2-s-t_m)-t_p*(4*m_X4-2*m_X2*(2*s+t_max+t_m)+s2+s*(t_max+t_m)+2*t_max*t_m)))/((t_max-m_d2)*(t_m-m_d2)*(t_p-m_d2)*(-m_d2-2*m_X2+s+t_max)*(-m_d2-2*m_X2+s+t_m)*(-m_d2-2*m_X2+s+t_p)))

    term1_min = -((8*(2*m_d2+m_X2)**2*np.sqrt((t_min-t_m)*(t_min-t_p))*(2*m_d6+m_d4*(6*m_X2-3*s-2*(t_min+t_m+t_p))+m_d2*(12*m_X4-4*m_X2*(3*s+t_min+t_m+t_p)+3*s2+2*s*(t_min+t_m+t_p)+2*t_p*(t_min+t_m)+2*t_min*t_m)+(2*m_X2-s)*(2*m_X2-s-t_min)*(2*m_X2-s-t_m)-t_p*(4*m_X4-2*m_X2*(2*s+t_min+t_m)+s2+s*(t_min+t_m)+2*t_min*t_m)))/((t_min-m_d2)*(t_m-m_d2)*(t_p-m_d2)*(-m_d2-2*m_X2+s+t_min)*(-m_d2-2*m_X2+s+t_m)*(-m_d2-2*m_X2+s+t_p)))

    term2_max = -(4*sqrt_fac_tmax*1/((m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)*(2*m_X2-s)))*(16*m_d8-16*m_d6*(t_m+t_p)-2*m_d4*(12*m_X4+4*m_X2*(-s+t_m+t_p)+s2-2*s*(t_m+t_p)-8*t_m*t_p)+2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p))-2*m_X2*(s*(t_m+t_p)-4*t_m*t_p)+s*(s*(t_m+t_p)-4*t_m*t_p))+2*m_X6*(t_m+t_p)-m_X4*(s*(t_m+t_p)+8*t_m*t_p)-2*s2*t_m*t_p)*(np.log(m_d2-t_max)-np.log(m_d2*(2*t_max-t_m-t_p)+2*np.sqrt(m_d2-t_m)*np.sqrt(m_d2-t_p)*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)-t_max*(t_m+t_p)+2*t_m*t_p))

    term2_min = -(4*sqrt_fac_tmin*1/((m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)*(2*m_X2-s)))*(16*m_d8-16*m_d6*(t_m+t_p)-2*m_d4*(12*m_X4+4*m_X2*(-s+t_m+t_p)+s2-2*s*(t_m+t_p)-8*t_m*t_p)+2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p))-2*m_X2*(s*(t_m+t_p)-4*t_m*t_p)+s*(s*(t_m+t_p)-4*t_m*t_p))+2*m_X6*(t_m+t_p)-m_X4*(s*(t_m+t_p)+8*t_m*t_p)-2*s2*t_m*t_p)*(np.log(m_d2-t_min)-np.log(m_d2*(2*t_min-t_m-t_p)+2*np.sqrt(m_d2-t_m)*np.sqrt(m_d2-t_p)*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)-t_min*(t_m+t_p)+2*t_m*t_p))

    term3_max = (4*sqrt_fac_tmax*(16*m_d8+16*m_d6*(6*m_X2-3*s-t_m-t_p)+2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p)+8*t_m*t_p)+2*m_d2*(34*m_X6-m_X4*(57*s+16*(t_m+t_p))+2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)-s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p))-24*m_X8+2*m_X6*(12*s+7*(t_m+t_p))-m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)+4*m_X2*s2*(2*s+t_m+t_p)-2*s2*(s+t_m)*(s+t_p))*(np.log(m_d2+2*m_X2-s-t_max)-np.log(2*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)*np.sqrt(m_d2+2*m_X2-s-t_m)*np.sqrt(m_d2+2*m_X2-s-t_p)+m_d2*(2*t_max-t_m-t_p)+m_X2*(4*t_max-2*(t_m+t_p))+t_p*(s-t_max+2*t_m)-2*s*t_max+s*t_m-t_max*t_m)))/((2*m_X2-s)*(m_d2+2*m_X2-s-t_m)**(3/2)*(m_d2+2*m_X2-s-t_p)**(3/2))

    term3_min = (4*sqrt_fac_tmin*(16*m_d8+16*m_d6*(6*m_X2-3*s-t_m-t_p)+2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p)+8*t_m*t_p)+2*m_d2*(34*m_X6-m_X4*(57*s+16*(t_m+t_p))+2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)-s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p))-24*m_X8+2*m_X6*(12*s+7*(t_m+t_p))-m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)+4*m_X2*s2*(2*s+t_m+t_p)-2*s2*(s+t_m)*(s+t_p))*(np.log(m_d2+2*m_X2-s-t_min)-np.log(2*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)*np.sqrt(m_d2+2*m_X2-s-t_m)*np.sqrt(m_d2+2*m_X2-s-t_p)+m_d2*(2*t_min-t_m-t_p)+m_X2*(4*t_min-2*(t_m+t_p))+t_p*(s-t_min+2*t_m)-2*s*t_min+s*t_m-t_min*t_m)))/((2*m_X2-s)*(m_d2+2*m_X2-s-t_m)**(3/2)*(m_d2+2*m_X2-s-t_p)**(3/2))

    term4_max = -((16*sqrt_fac_tmax*np.log(2*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)+2*t_max-t_m-t_p)))
    term4_min = -((16*sqrt_fac_tmin*np.log(2*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)+2*t_min-t_m-t_p)))

    return vert*(term1_max + term2_max + term3_max + term4_max - term1_min - term2_min - term3_min - term4_min).real

@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s(s, E1, E2, E3, p1, p3, m_d, m_X, s12_min, s12_max, s34_min, s34_max, vert):
    p12 = p1*p1
    p32 = p3*p3
    # Anton: a,b,c definition in a*cos^2 + b*cos + c = 0 in integrand
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2))*(s-2.*E3*(E1+E2))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    # Anton: ct_p, ct_m solutions of a*cos^2 + b*cos + c = 0 for cos
    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)
    # Anton: R_theta integration region {-1 <= cos <= 1 | c_p <= cos <= c_m}
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    in_res = (ct_max > ct_min)

    # Anton: return zero for integral if it is not inside defined region.
    # x = [0,0,0], x[[True,False,True]] = [1,2,3] => x = [1,0,3]
    t_int = np.zeros(s.size)
    t_int[in_res] = ker_C_n_XX_dd_s_t_integral_2(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, vert)
    return t_int

# 3 4 -> 1 2 <=> X X -> d d
@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd(x, m_d, m_X, k_d, k_phi, T_d, xi_d, xi_X, vert):
    """
    Anton: Seems like E1 <--> E3, E2 <--> E4 compared to article.
    Think it is because we now do XX --> dd instead of dd --> XX, 
    where 1,2 = d,d and 3,4 = X,X as someone once wrote in comment 
    above the function (3 4 -> 1 2 <=> phi phi -> d d). 
    """
    log_E3_min = log(m_X*offset)
    log_E3_max = log(max((max_exp_arg + xi_X)*T_d, 1e1*m_X))
    E3 = np.exp(np.fmin(log_E3_min * (1.-x[:,0]) + log_E3_max * x[:,0], 6e2))

    E4_min = np.fmax(2.*m_d-E3, m_X*offset)
    log_E4_min = np.log(E4_min)
    log_E4_max = np.log(np.fmax(1e1*E4_min, (max_exp_arg + xi_X)*T_d))
    E4 = np.exp(np.fmin(log_E4_min * (1.-x[:,1]) + log_E4_max * x[:,1], 6e2))

    log_E1_min = np.log(m_d*offset)
    log_E1_max = np.log(np.fmax(E3+E4-m_d, m_d*offset))
    E1 = np.exp(np.fmin(log_E1_min * (1.-x[:,2]) + log_E1_max * x[:,2], 6e2))
    E2 = E3 + E4 - E1

    exp_arg_1 = E1/T_d - xi_d
    exp_arg_2 = E2/T_d - xi_d
    exp_arg_3 = E3/T_d - xi_X
    exp_arg_4 = E4/T_d - xi_X
    exp_1 = np.exp(np.fmin(-exp_arg_1, max_exp_arg))
    exp_2 = np.exp(np.fmin(-exp_arg_2, max_exp_arg))
    exp_3 = np.exp(np.fmin(-exp_arg_3, max_exp_arg))
    exp_4 = np.exp(np.fmin(-exp_arg_4, max_exp_arg))
    f1 = exp_1/(1. + k_d*exp_1)
    f2 = exp_2/(1. + k_d*exp_2)
    f3 = exp_3/(1. + k_phi*exp_3)
    f4 = exp_4/(1. + k_phi*exp_4)
    dist = f3*f4*(1.-k_d*f1)*(1.-k_d*f2)
    # dist = f1*f2*(1.-k_phi*f3)*(1.-k_phi*f4)

    # Anton: Three-momentum p^2 = E^2 - m^2 = (E - m)*(E + m)
    p1 = np.sqrt(np.fmax((E1-m_d)*(E1+m_d), 1e-200))
    p2 = np.sqrt(np.fmax((E2-m_d)*(E2+m_d), 1e-200))
    p3 = np.sqrt(np.fmax((E3-m_X)*(E3+m_X), 1e-200))
    p4 = np.sqrt(np.fmax((E4-m_X)*(E4+m_X), 1e-200))

    s12_min = np.fmax(2.*m_d*m_d+2.*E1*(E2-p1*p2/E1), 2.*m_d*m_d)
    s12_max = 2.*m_d*m_d+2.*E1*E2+2.*p1*p2
    s34_min = np.fmax(2.*m_X*m_X+2.*E3*(E4-p3*p4/E3), 2.*m_X*m_X)
    s34_max = 2.*m_X*m_X+2.*E3*E4+2.*p3*p4
    log_s_min = np.log(np.fmax(np.fmax(s12_min, s34_min), 1e-200))
    log_s_max = np.log(np.fmax(np.fmin(s12_max, s34_max), 1e-200))
    s = np.exp(np.fmin(log_s_min * (1.-x[:,3]) + log_s_max * x[:,3], 6e2))

    ker_s = ker_C_n_XX_dd_s(s, E1, E2, E3, p1, p3, m_d, m_X, s12_min, s12_max, s34_min, s34_max, vert)

    jac = E3*(log_E3_max-log_E3_min)*E4*(log_E4_max-log_E4_min)*E1*(log_E1_max-log_E1_min)*s*(log_s_max-log_s_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    return res

# type == -1: only phi phi -> d d, type == 0: both reactions, type == 1: only d d -> phi phi, type == 2: (phi phi -> d d, d d -> phi phi)
def C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, vert, type = 0):
    if m_X/T_d - xi_X > spin_stat_irr: # spin-statistics irrelevant here
        th_avg_s_v = th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert)
        if th_avg_s_v <= 0.:
            if type == 2:
                return np.array([0., 0.])
            return 0.
        if type == 0:
            chem_eq_fac = exp(2.*xi_d) - exp(2.*xi_X)
        elif type == -1:
            chem_eq_fac = - exp(2.*xi_X)
        elif type == 1:
            chem_eq_fac = exp(2.*xi_d)
        elif type == 2:
            return np.array([- exp(2.*xi_X), exp(2.*xi_d)])*th_avg_s_v
        return chem_eq_fac*th_avg_s_v

    if type == 0:
        chem_eq_fac = exp(2.*(xi_d-xi_X)) - 1.
    elif type == -1:
        chem_eq_fac = -1.
    elif type == 1:
        chem_eq_fac = exp(2.*(xi_d-xi_X))

    # Send arrays in batches
    @vegas.batchintegrand
    def kernel(x):
        return ker_C_n_XX_dd(x, m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, vert)

    """
    Anton: Order of integration in analytic expression: E1, E2, E3, s. 
    Implementation reads the order: E3, E4, E1, where E2 has been eliminated instead of E4.  
    Seems like a change of variables has been done, see inside ker_C_n_XX_dd function. 
    Seemingly, 

    x_i = ln(E_i / E_i_min) / ln(E_i_max / E_i_min) where E_i_min/max is lower/upper integration bound of E_i. 
    s' = ln(s / s_min) / ln(s_max / s_min) where s_min/max is lower/upper integration bound of s.

    Then {x_i, s' in [0, 1]}, and 
    jacobian = E1*(log_E1_max - log_E1_min)*E2*(log_E2_max - log_E2_min)*E3*(log_E3_max - log_E3_min)*s*(log_s_max - log_s_min)
    """

    # Monte-Carlo integration of the 4 integrals from 0 to 1 
    integ = vegas.Integrator(4 * [[0., 1.]])        # Sends one and one value at a time 
    result = integ(kernel, nitn=10, neval=2e5)
    # if result.mean != 0.:
    #     print("Vegas error pp dd: ", result.sdev/fabs(result.mean), result.mean, result.Q)
    # print("pp dd", result.mean*chem_eq_fac/(256.*(pi**6.)), (exp(2.*xi_d)-exp(2.*xi_X))*th_avg_sigma_v_XX_dd(T_d, m_d, m_phi, vert))

    if type == 2:
        return np.array([-1., exp(2.*(xi_d-xi_X))])*result.mean/(256.*(pi**6.))
    
    return result.mean*chem_eq_fac/(256.*(pi**6.))


@nb.jit(nopython=True, cache=True)
def sigma_pp_dd(s, m_d, m_phi, vert):
    """
    Anton: Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we must integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = -(p1cm - p3m)^2 = -(p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds 
    t_upper = -(p1cm - p3cm)^2
    t_lower = -(p1cm + p3cm)^2
    s = (p1/3 + p2/4)^2 = (E1/3 + E2/4)^2 = 4E1/3^2, since m1 = m2, m3 = m4 in this case
    => p1cm = sqrt(1/4*s - m1^2), p3cm = sqrt(1/4*s - m3^2)
    Generically with m1 != m2, m3 != m4, p1/3cm = sqrt(1/(4*s) * [s^2 - 2*s(m1/3^2 + m2/4^2) + (m1/3^2 - m2/4^2)^2])
    Cross-section:
    sigma = H(E_cm - m3 - m4)/(16*pi*[s^2 - 2*s(m1^2 + m2^2) + (m1^2 - m2^2)^2]) * int_{t_lower}^{t_upper} dt |M|^2
    """
    m_d2 = m_d*m_d
    m_phi2 = m_phi*m_phi
    m_phi4 = m_phi2*m_phi2
    if s <= 4.*m_phi2 or s <= 4.*m_d2:
        return 0.
    p3cm = sqrt(0.25*s - m_phi2)
    p1cm = sqrt(0.25*s - m_d2)
    t0 = -((p1cm-p3cm)**2.)
    t1 = -((p1cm+p3cm)**2.)

    int0 = 4*(t0-m_d2) + m_phi4*(1./(m_d2-t0)+1./(m_d2+2.*m_phi2-s-t0))
    int1 = 4*(t1-m_d2) + m_phi4*(1./(m_d2-t1)+1./(m_d2+2.*m_phi2-s-t1))
    log_part = (6.*m_phi4-4.*m_phi2*s+s*s)*log((m_d2+2.*m_phi2-s-t0)*(m_d2-t1)/((m_d2-t0)*(m_d2+2.*m_phi2-s-t1)))/(2.*m_phi2-s)

    return vert*(int0-int1+log_part)/(8.*pi*s*(4.*m_phi2-s))

# X X -> d d
@nb.jit(nopython=True, cache=True)
def sigma_XX_dd(s, m_d, m_X, vert):
    """
    Anton: Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we must integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2
      = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 
    sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum p1/3cm = sqrt(E1/3cm^2 - m1/2^2) for integration bounds. 
    In this case, m1 = m3, m2 = m4.
    Heavysides from demanding positive p_cm^2: 
    H(1/(4*s)*[(s - m1/3 - m2/4)^2 - 4*m1/3^2*m2/4^2]) = H((s - m1/3 - m2/4)^2 - 4*m1/3^2*m2/4^2)
    = H(s - m1/3 - m2/4 - 2*m1/3*m2/4)
    = H(s - (m1/3 + m2/4)^2)
    = H(E_cm - m1/3 - m2/4)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(16*pi*[s^2 - 2*s(m1^2 + m2^2) + (m1^2 - m2^2)^2]) 
          * int_{t_lower}^{t_upper} dt |M|^2
    """
    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    m_d4 = m_d2*m_d2
    m_X4 = m_X2*m_X2

    # Sigma zero when Heavyside violated, obtain values everywhere else
    cross_section = np.zeros(s.size)
    # Area where the three-momenta is defined.
    res = np.logical_and(s > 4*m_d2, s > 4*m_X2)

    s_res = s[res]
    s2_res = s_res*s_res

    # Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s_res - m_d2)
    p3cm = np.sqrt(0.25*s_res - m_X2)

    # Upper and lower integration bound 
    # Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    # t-integrated squared matrix elements
    # imaginary parts from upper - lower will cancel
    int_t_M2_upper = 8*vert*((2*m_d2+m_X2)**2/(-m_d2-2*m_X2+s_res+t_upper) + (2*m_d2+m_X2)**2/(t_upper-m_d2) + ((-8*m_d4+4*m_d2*(s_res-2*m_X2)+4*m_X4+s2_res)*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s_res+t_upper)))/(2*m_X2-s_res) - 2*t_upper)

    int_t_M2_lower = 8*vert*((2*m_d2+m_X2)**2/(-m_d2-2*m_X2+s_res+t_lower) + (2*m_d2+m_X2)**2/(t_lower-m_d2) + ((-8*m_d4+4*m_d2*(s_res-2*m_X2)+4*m_X4+s2_res)*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s_res+t_lower)))/(2*m_X2-s_res) - 2*t_lower)

    cross_section[res] = ((int_t_M2_upper - int_t_M2_lower).real / (16.*np.pi*s_res*(s_res - 4*m_X2)))
    return cross_section

def ker_th_avg_sigma_v_XX_dd(log_s, T_d, m_d, m_phi, vert):
    s = exp(log_s)
    sqrt_s = sqrt(s)
    sigma = sigma_XX_dd(s, m_d, m_phi, vert)
    return s*sigma*(s-4.*m_phi*m_phi)*sqrt_s*kn(1, sqrt_s/T_d)

# only \int d^3 p3 d^3 p4 sigma v exp(-(E3+E4)/T)/(2 pi)^6
def th_avg_sigma_v_XX_dd(T_d, m_d, m_phi, vert):
    s_min = max(4.*m_d*m_d, 4.*m_phi*m_phi)
    s_max = (5e2*T_d)**2.
    if s_max <= s_min:
        return 0.

    res, err = quad(ker_th_avg_sigma_v_XX_dd, log(s_min), log(s_max), args=(T_d, m_d, m_phi, vert), epsabs=0., epsrel=rtol_int)

    return res*T_d/(32.*(pi**4.))
    # return res/(8.*(m_phi**4.)*T_d*(kn(2, m_phi/T_d)**2.))

@nb.jit(nopython=True, cache=True)
def ker_C_34_12_s_t_integral(ct_min, ct_max, ct_p, ct_m, s, E1, E3, p1, p3, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, res_sub):
    sqrt_arg_min = (ct_m-ct_min)/(ct_m-ct_p)
    sqrt_arg_max = (ct_m-ct_max)/(ct_m-ct_p)
    sqrt_fac_min = np.sqrt(sqrt_arg_min)
    sqrt_fac_max = np.sqrt(sqrt_arg_max)
    x0 = -2.*np.arcsin(sqrt_fac_min)
    x1 = -2.*np.arcsin(sqrt_fac_max)

    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    m42 = m4*m4
    m13 = m12*m1
    m23 = m22*m2
    m33 = m32*m3
    m43 = m42*m4
    m_Gamma_phi = np.sqrt(m_Gamma_phi2)

    t_m = m12 + m32 - 2.*E1*(E3-p1*p3*ct_m/E1)
    t_p = m12 + m32 - 2.*E1*(E3-p1*p3*ct_p/E1)
    u_m = m22 + m42 - s + 2.*E1*(E3-p1*p3*ct_m/E1)
    u_p = m22 + m42 - s + 2.*E1*(E3-p1*p3*ct_p/E1)

    i = complex(0., 1.)
    i_m_Gamma_phi = i*m_Gamma_phi
    sqrt_t_p = np.sqrt(m_phi2-t_p-i_m_Gamma_phi)
    sqrt_t_m = np.sqrt(m_phi2-t_m-i_m_Gamma_phi)
    sqrt_u_p = np.sqrt(u_p-m_phi2-i_m_Gamma_phi)
    sqrt_u_m = np.sqrt(u_m-m_phi2-i_m_Gamma_phi)
    denom0 = np.sqrt(1.-sqrt_arg_min)
    denom1 = np.sqrt(1.-sqrt_arg_max)
    atan0_t = np.arctan(-np.divide(sqrt_fac_min*sqrt_t_p, sqrt_t_m*denom0))
    atan1_t = np.arctan(-np.divide(sqrt_fac_max*sqrt_t_p, sqrt_t_m*denom1))
    atan_t_diff = atan1_t - atan0_t
    atan0_u = np.arctan(-np.divide(sqrt_fac_min*sqrt_u_p, sqrt_u_m*denom0))
    atan1_u = np.arctan(-np.divide(sqrt_fac_max*sqrt_u_p, sqrt_u_m*denom1))
    atan_u_diff = atan1_u - atan0_u

    s_prop = 1./((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)

    if res_sub:
        ss = (x1-x0)*((m1+m2)*(m1+m2)-s)*((m3+m4)*(m3+m4)-s)*(s-m_phi2)*(s-m_phi2)*s_prop*s_prop
    else:
        ss = (x1-x0)*((m1+m2)*(m1+m2)-s)*((m3+m4)*(m3+m4)-s)*s_prop
    tt_comp = -i*(m12+2.*m1*m3+m32-m_phi2+i_m_Gamma_phi)*(m22+2.*m2*m4+m42-m_phi2+i_m_Gamma_phi)*atan_t_diff/(m_Gamma_phi*sqrt_t_p*sqrt_t_m)
    tt = x1 - x0 + 2.*tt_comp.real
    uu_comp = -i*(m22+2.*m2*m3+m32-m_phi2-i_m_Gamma_phi)*(m12+2.*m1*m4+m42-m_phi2-i_m_Gamma_phi)*atan_u_diff/(m_Gamma_phi*sqrt_u_p*sqrt_u_m)
    uu = x1 - x0 + 2.*uu_comp.real
    st_comp = (s-m_phi2+i_m_Gamma_phi)*(m23*m3+m13*m4+m22*m3*(m3+m4)+m12*m4*(m2+m3+m4)-s*(m3*m4+m_phi2-i_m_Gamma_phi)+
     m2*(m33+m32*m4-m4*m_phi2+m4*i_m_Gamma_phi-m3*(m_phi2+s-i_m_Gamma_phi))+
     m1*(m22*m3+m2*(m32+2.*m3*m4+m42-s)+m3*(m42-m_phi2+i_m_Gamma_phi)+m4*(m42-m_phi2-s+i_m_Gamma_phi)))*\
     atan_t_diff/(sqrt_t_p*sqrt_t_m)
    st = ((s-m_phi2)*((m1+m2)*(m3+m4)+s)*(x1-x0) + 2.*st_comp.real)*s_prop
    su_comp = (m_phi2-s+i_m_Gamma_phi)*(m13*m3+m23*m4+m22*m4*(m3+m4)+m12*m3*(m2+m3+m4)-s*m3*m4+m2*m3*(m42-m_phi2-i_m_Gamma_phi)-
     m_phi2*s-s*i_m_Gamma_phi-m2*m4*(-m42+m_phi2+s+i_m_Gamma_phi)+
     m1*(m33+m4*(m22+m32)+m2*(m32+2.*m3*m4+m42-s)-m4*m_phi2-m4*i_m_Gamma_phi-m3*(m_phi2+s+i_m_Gamma_phi)))*\
     atan_u_diff/(sqrt_u_p*sqrt_u_m)
    su = ((s-m_phi2)*((m1+m2)*(m3+m4)+s)*(x1-x0) + 2.*su_comp.real)*s_prop
    tu_comp_t = -(m23*m3+m13*m4-m32*m42+m_phi2*(m32+m42)-m_phi2*m_phi2-m3*m4*s-m_phi2*s+m22*(m3*m4+m_phi2-i_m_Gamma_phi)+
     m12*(-m22+m4*(m3-m2)+m_phi2-i_m_Gamma_phi)+i_m_Gamma_phi*(2.*m_phi2+s-m32-m42)+m_Gamma_phi2+
     m2*(m33-m32*m4+m4*m_phi2-m4*i_m_Gamma_phi-m3*(m_phi2+s-i_m_Gamma_phi))-
     m1*(m22*m3-m2*((m3-m4)*(m3-m4)-s)+m4*(-m42+m_phi2+s-i_m_Gamma_phi)+m3*(m42-m_phi2+i_m_Gamma_phi)))*\
     atan_t_diff/((m12+m22+m32+m42-2.*m_phi2-s)*sqrt_t_p*sqrt_t_m)
    tu_comp_u = (m13*m3+m23*m4-m32*m42+m_phi2*(m32+m42)-m_phi2*m_phi2-m3*m4*s-m_phi2*s+m22*(m3*m4+m_phi2+i_m_Gamma_phi)+
     m12*(-m22+m3*(m4-m2)+m_phi2+i_m_Gamma_phi)+i_m_Gamma_phi*(m32+m42-2.*m_phi2-s)+m_Gamma_phi2+
     m2*(-m3*m42+m3*m_phi2+m3*i_m_Gamma_phi+m4*(m42-m_phi2-s-i_m_Gamma_phi))-
     m1*(-m33-m2*((m3-m4)*(m3-m4)-s)+m3*(m_phi2+s+i_m_Gamma_phi)+m4*(m22+m32-m_phi2-i_m_Gamma_phi)))*\
     atan_u_diff/((m12+m22+m32+m42-2.*m_phi2-s)*sqrt_u_p*sqrt_u_m)
    tu = x1 - x0 + 2.*tu_comp_t.real + 2.*tu_comp_u.real

    # for i in range(x0.size):
    #     def kernel_S_123_x(x):
    #         ct = 0.5*(ct_m[i]+ct_p[i]+(ct_m[i]-ct_p[i])*cos(x))
    #         t = m12 + m32 - 2.*E3[i]*(E1[i] - p1[i]*p3[i]*ct/E3[i])
    #
    #         return scalar_mediator.M2_gen(s[i], t, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=True)
    #     res, err = quad(kernel_S_123_x, x0[i], x1[i], epsabs=0., epsrel=rtol_int)
    #     if fabs(4.*vert*(ss[i]+tt[i]+uu[i]+st[i]+su[i]+tu[i])/res - 1.) > 1e-4:
    #         print(4.*vert*(ss[i]+tt[i]+uu[i]+st[i]+su[i]+tu[i]), res)

    return 4.*vert*(ss+tt+uu+st+su+tu)

@nb.jit(nopython=True, cache=True)
def ker_C_34_12_s(s, E1, E2, E3, p1, p2, p3, s12_min, s12_max, s34_min, s34_max, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, res_sub):
    p12 = p1*p1
    p32 = p3*p3
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2)+(m1-m2)*(m1+m2))*(s-2.*E3*(E1+E2)+(m3-m4)*(m3+m4))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))
    ct_p = (-b+sqrt_fac)/(2.*a)
    ct_m = (-b-sqrt_fac)/(2.*a)
    # if fabs(ct_m - 1.) < 1e-12:
    #     ct_m = 1.
    # if fabs(ct_p + 1.) < 1e-12:
    #     ct_p = -1.
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    in_res = (ct_max > ct_min)

    t_int = np.zeros(s.size)
    t_int[in_res] = ker_C_34_12_s_t_integral(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2[in_res], res_sub)/np.sqrt(-a[in_res])
    return t_int

@nb.jit(nopython=True, cache=True)
def ker_C_34_12(x, log_s_min, log_s_max, type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_phi2, m_Gamma_phi2, res_sub, thermal_width):
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    m42 = m4*m4

    s = np.exp(np.fmin(log_s_min * (1.-x[:,0]) + log_s_max * x[:,0], 6e2))

    log_E1_min = log(max(m1*offset, 1e-200))
    log_E1_max = log(max((max_exp_arg + xi1)*T1, 1e1*m1))
    E1 = np.exp(np.fmin(log_E1_min * (1.-x[:,1]) + log_E1_max * x[:,1], 6e2))
    # if E1 <= m1:
    #     return 0. # otherwise problems in computation (division by p1)
    p1 = np.sqrt(np.fmax((E1-m1)*(E1+m1), 1e-200))

    sqrt_fac_2 = np.sqrt(np.fmax(s*s-2.*(m12+m22)*s+((m1+m2)*(m1-m2))**2., 0.))
    E2_min = np.fmax((E1*(s-m12-m22)-p1*sqrt_fac_2)/(2.*m12), max(m2*offset, 1e-200))
    E2_max = (E1*(s-m12-m22)+p1*sqrt_fac_2)/(2.*m12)
    log_E2_min = np.log(E2_min)
    log_E2_max = np.log(E2_max)
    E2 = np.exp(log_E2_min * (1.-x[:,2]) + log_E2_max * x[:,2])
    p2 = np.sqrt(np.fmax((E2-m2)*(E2+m2), 1e-200))

    E12 = E1+E2
    E122 = E12*E12
    sqrt_fac_3 = np.sqrt(np.fmax((E122-s)*(s*s-2.*(m32+m42)*s+((m3+m4)*(m3-m4))**2.), 0.))
    E3_min = np.fmax((E12*(s+m32-m42)-sqrt_fac_3)/(2.*s), max(m3*offset, 1e-200))
    E3_max = (E12*(s+m32-m42)+sqrt_fac_3)/(2.*s)
    log_E3_min = np.log(E3_min)
    log_E3_max = np.log(E3_max)
    E3 = np.exp(np.fmin(log_E3_min * (1.-x[:,3]) + log_E3_max * x[:,3], 6e2))
    p3 = np.sqrt(np.fmax((E3-m3)*(E3+m3), 1e-200))

    E4 = E12 - E3
    p4 = np.sqrt(np.fmax((E4-m4)*(E4+m4), 1e-200))

    exp_arg_1 = E1/T1 - xi1
    exp_arg_2 = E2/T2 - xi2
    exp_arg_3 = E3/T3 - xi3
    exp_arg_4 = E4/T4 - xi4
    exp_1 = np.exp(np.fmin(-exp_arg_1, max_exp_arg))
    exp_2 = np.exp(np.fmin(-exp_arg_2, max_exp_arg))
    exp_3 = np.exp(np.fmin(-exp_arg_3, max_exp_arg))
    exp_4 = np.exp(np.fmin(-exp_arg_4, max_exp_arg))
    f1 = exp_1/(1. + k1*exp_1)
    f2 = exp_2/(1. + k2*exp_2)
    f3 = exp_3/(1. + k3*exp_3)
    f4 = exp_4/(1. + k4*exp_4)
    dist_FW = nFW*f3*f4*(1.-k1*f1)*(1.-k2*f2)
    dist_BW = nBW*f1*f2*(1.-k3*f3)*(1.-k4*f4)
    if type == 0.:
        Etype = np.ones(E1.size)
    elif type == 1.:
        Etype = E1
    elif type == 2.:
        Etype = E2
    elif type == 3.:
        Etype = E3
    elif type == 4.:
        Etype = E4
    else:
        Etype = E12
    dist = Etype*(dist_FW+dist_BW)

    if thermal_width:
        m_phi = sqrt(m_phi2)
        sqrt_arg = (m_phi2-4.*m3*m3)*((E1+E2)**2.-m_phi2)
        sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 1e-200))
        E3p = 0.5*(E12+sqrt_fac/m_phi)
        E3m = 0.5*(E12-sqrt_fac/m_phi)
        exp_3p_xi = np.exp(np.fmin(xi3-E3p/T3, max_exp_arg))
        exp_3m_xi = np.exp(np.fmin(xi3-E3m/T3, max_exp_arg))
        E3_integral = sqrt_fac/(T3*m_phi) + np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))
        m_Gamma_phi_T = sqrt(m_Gamma_phi2)*(1.+m_phi*T3*np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))/sqrt_fac)
        m_Gamma_phi_T2 = m_Gamma_phi_T*m_Gamma_phi_T
    else:
        m_Gamma_phi_T2 = m_Gamma_phi2*np.ones(s.size)

    s12_min = m12+m22+2.*E1*(E2-p1*p2/E1)
    s12_max = m12+m22+2.*E1*E2+2.*p1*p2
    s34_min = m32+m42+2.*E3*(E4-p3*p4/E3)
    s34_max = m32+m42+2.*E3*E4+2.*p3*p4
    ker_s = ker_C_34_12_s(s, E1, E2, E3, p1, p2, p3, s12_min, s12_max, s34_min, s34_max, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi_T2, res_sub)

    jac = E1*(log_E1_max-log_E1_min)*E2*(log_E2_max-log_E2_min)*E3*(log_E3_max-log_E3_min)*s*(log_s_max-log_s_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    return res

# 3 4 -> 1 2 (all neutrinos); nFW (nBW): # of particle occurence in final - initial state for forward 3 4 -> 1 2 (backward 1 2 -> 3 4) reaction
# type indicates if for n (0) or rho (1 for E1, 2 for E2, 3 for E3, 4 for E4, 12 for E1+E2 = E3+E4)
# note that when using thermal width it is assumed that m3 = m4 = md, T3 = T4 = Td, xi3 = xi4 = xid, xi_phi = 2 xi_d
# and 3, 4 are fermions, phi is boson
def C_34_12(type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_phi2, m_Gamma_phi2, res_sub = False, thermal_width = True):
    s_min = max((m1+m2)**2., (m3+m4)**2.)*offset # to prevent accuracy problems
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)
    p1_max = sqrt((E1_max-m1)*(E1_max+m1))
    E2_max = max((max_exp_arg + xi2)*T2, 1e1*m2)
    p2_max = sqrt((E2_max-m2)*(E2_max+m2))
    E3_max = max((max_exp_arg + xi3)*T3, 1e1*m3)
    p3_max = sqrt((E3_max-m3)*(E3_max+m3))
    E4_max = max((max_exp_arg + xi4)*T4, 1e1*m4)
    p4_max = sqrt((E4_max-m4)*(E4_max+m4))
    s12_max = m1*m1+m2*m2+2.*E1_max*E2_max+2.*p1_max*p2_max
    s34_max = m3*m3+m4*m4+2.*E3_max*E4_max+2.*p3_max*p4_max
    s_max = max(s12_max, s34_max)
    s_vals = np.sort(np.array([s_min, s_max, m_phi2-fac_res_width*sqrt(m_Gamma_phi2), m_phi2, m_phi2+fac_res_width*sqrt(m_Gamma_phi2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    np.seterr(divide='ignore')
    for i in range(len(s_vals)-1):
        @vegas.batchintegrand
        def kernel(x):
            return ker_C_34_12(x, log(s_vals[i]), log(s_vals[i+1]), type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_phi2, m_Gamma_phi2, res_sub, thermal_width)
        integ = vegas.Integrator(4 * [[0., 1.]])
        result = integ(kernel, nitn=10, neval=1e5)
        # print(result.summary())
        # if result.mean != 0.:
        #     print("Vegas error 34 12: ", result.sdev/fabs(result.mean), result.mean/(256.*(pi**6.)), result.Q)
        res += result.mean
    np.seterr(divide='warn')
    # print("34 12:", res/(256.*(pi**6.)), (th_avg_sigma_v_33_11(m3, m4, m1, T1, vert, m_phi2, m_Gamma_phi2)*(nBW*exp(xi1+xi2))))

    return res/(256.*(pi**6.))

def ker_th_avg_sigma_v_33_11(log_s, m1, m2, m3, T, vert, m_phi2, m_Gamma_phi2, res_sub):
    s = exp(log_s)
    sqrt_s = sqrt(s)
    sigma = scalar_mediator.sigma_gen(s, m3, m3, m1, m2, vert, m_phi2, m_Gamma_phi2, sub=res_sub)
    # print(log_s, s*sigma*(s-4.*m1*m1)*sqrt_s*kn(1, sqrt_s/T3))
    return s*sigma*(s-4.*m3*m3)*sqrt_s*kn(1, sqrt_s/T)

# only \int d^3 p3 d^3 p4 sigma v exp(-(E3+E4)/T)/(2 pi)^6
def th_avg_sigma_v_33_11(m1, m2, m3, T, vert, m_phi2, m_Gamma_phi2, res_sub = True):
    s_min = max((m1+m2)*(m1+m2), 4.*m3*m3)*offset
    s_max = (1e3*T)**2.
    if s_max <= s_min:
        return 0.
    s_vals = np.sort(np.array([s_min, s_max, m_phi2-fac_res_width*sqrt(m_Gamma_phi2), m_phi2, m_phi2+fac_res_width*sqrt(m_Gamma_phi2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    for i in range(len(s_vals)-1):
        cur_res, err = quad(ker_th_avg_sigma_v_33_11, log(s_vals[i]), log(s_vals[i+1]), args=(m1, m2, m3, T, vert, m_phi2, m_Gamma_phi2, res_sub), epsabs=0., epsrel=rtol_int, limit=100)
        res += cur_res

    return res*T/(32.*(pi**4.))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    m_d = 1e-5      # 1e-5 GeV = 10 kev 
    m_a = 0.
    m_X = 3*m_d
    sin2_2th = 1e-12
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    y = 2e-3

    vert = y**4 * np.cos(th)**8
    # vert = 1

    T = 0.6510550394714374
    xi_d = -8.551301127056323
    xi_X = 0.

    ########################################################################
    # Mostly plot things to check 
    x = np.linspace(0, 1, int(1e3))    # x = ln(s/s_min) / ln(s_max/s_min)
    T_d = T
    # T_d = 1

    # Switched from E1, E2 --> E3, E4 to E3, E4 --> E1, E2 (dd --> XX to XX --> dd)
    # Anton: Treated as E1 in article 
    log_E3_min = np.log(m_X*offset)
    log_E3_max = np.log(max((max_exp_arg + xi_X)*T_d, 1e1*m_X))
    E3_ = np.exp(np.fmin(log_E3_min * (1.-x) + log_E3_max * x, 6e2))
    # Anton: This is treated as E2 in the article 
    E4_min = np.fmax(2.*m_d - E3_, m_X*offset)
    log_E4_min = np.log(E4_min)
    log_E4_max = np.log(np.fmax(1e1*E4_min, (max_exp_arg + xi_X)*T_d))
    E4_ = np.exp(np.fmin(log_E4_min * (1.-x) + log_E4_max * x, 6e2))
    # Anton: This is treated as E3 in article 
    log_E1_min = np.log(m_d*offset)
    log_E1_max = np.log(np.fmax(E3_ + E4_ - m_d, m_d*offset))
    E1_ = np.exp(np.fmin(log_E1_min * (1.-x) + log_E1_max * x, 6e2))
    # Anton: Treated as E4 in article 
    E2_ = E3_ + E4_ - E1_

    # n_plots = 5
    # for kidx, k in enumerate(np.linspace(0, 1, n_plots)):
    #     col = 5
    #     row = 5
        # fig = plt.figure()
    #     for iidx, i in enumerate(np.linspace(0, 1, col)):
    #         for jidx, j in enumerate(np.linspace(0, 1, row)):
    # ax = fig.add_subplot(int(row), int(col), int(row)*iidx+jidx+1)
    fig = plt.figure()
    ax = fig.add_subplot()
    E3 = E3_[int((x.size-1)*0.5)]
    E4 = E4_[int((x.size-1)*0.2)]
    E1 = E1_[int((x.size-1)*0.1)]
    E2 = E4 + E3 - E1

    p1 = np.sqrt(np.fmax((E1 - m_d)*(E1 + m_d), 1e-200))
    p2 = np.sqrt(np.fmax((E2 - m_d)*(E2 + m_d), 1e-200))
    p3 = np.sqrt(np.fmax((E3 - m_X)*(E3 + m_X), 1e-200))
    p4 = np.sqrt(np.fmax((E4 - m_X)*(E4 + m_X), 1e-200))

    # Kinematical region for s: s_min/max = (E + E')^2 - (p +- p')^2.
    # Avoid catastrophic cancellation by a^2 - b^2 = (a+b)*(a-b)
    s12_min = (E1 + E2 + p1 + p2)*(E1 + E2 - p1 - p2)
    s12_max = (E1 + E2 + p1 - p2)*(E1 + E2 - p1 + p2)
    s34_min = (E3 + E4 + p3 + p4)*(E3 + E4 - p3 - p4)
    s34_max = (E3 + E4 + p3 - p4)*(E3 + E4 - p4 + p4)
    log_s_min = np.log(np.fmax(np.fmax(s12_min, s34_min), 1e-200))
    log_s_max = np.log(np.fmax(np.fmin(s12_max, s34_max), 1e-200))
    s = np.exp(np.fmin(log_s_min * (1.-x) + log_s_max * x, 6e2))

    # print(s12_min, s12_max, s34_min, s34_max)

    # Something weird about s and the limits varying - s is multivalued in x 
    # s1 = np.exp(np.log(s[0]) * (1.-x) + np.log(np.max(s)) * x)

    # plt.plot(x, E1, 'tab:blue')
    # plt.plot(x, E2, 'r--')
    # plt.plot(x, E3, 'k--')
    # plt.plot(x, E4, 'g-.')
    # plt.plot(x,s)
    # plt.plot(x,s1)
    # plt.show()

    a = np.fmin(-4.*p3*p3*((E1 + E2)*(E1 + E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s - 2.*E1*(E1 + E2))*(s - 2.*E3*(E3 + E4))
    sqrt_arg = 4.*(p3*p3/(p1*p1))*(s - s12_min)*(s - s12_max)*(s - s34_min)*(s - s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    # Anton: ct_p, ct_m solutions of a*cos^2 + b*cos + c = 0 for cos
    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)

    # print(-sqrt_fac,a)

    # Anton: R_theta integration region {-1 <= cos <= 1 | c_p <= cos <= c_m}. fmin = element-wise
    # Take ct_p if -1 <= ct_p <= 1, else -1
    # Take ct_m if ct_min <= ct_m <= 1. If ct_m > 1, take 1, if 1 < ct_m < ct_min, take ct_min
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)   

    # Consider whether or not integration region is ill-defined, e.g. -1 < c_m < c_p < 1
    in_res = (ct_max > ct_min)
    # Pick one E_1, E_3 value to plot for
    # print(f'E_1: {E1[index_1]:.3e}, E_3: {E3[index_3]:.3e}')

    time1 = time.time()
    ker_C_n_XX_dd_s_t_integral_val = ker_C_n_XX_dd_s_t_integral_2(ct_min=ct_min[in_res], ct_max=ct_max[in_res], ct_p=ct_p[in_res], ct_m=ct_m[in_res], a=a, s=s[in_res], E1=E1, E3=E3, p1=p1, p3=p3, m_d=m_d, m_X=m_X, vert=vert)
    print(f'ker_C_n_XX_dd_s_t_integral ran in {time.time()-time1}s')
    # print(ker_C_n_XX_dd_s_t_integral_val)

    # x = ln(s/s_min) / ln(s_max/s_min)
    ax.plot(s[in_res], ker_C_n_XX_dd_s_t_integral_val, 'r')
    ax.set_xscale('log')
            
    # fig.tight_layout()
    plt.show()

    from matplotlib import collections  as mc

    fig, ax = plt.subplots()

    sigma = sigma_XX_dd(s=s, m_d=m_d, m_X=m_X, vert=vert)

    # Color lines based on which index element has, for debugging purposes
    # lines = [np.column_stack([[s[i], s[i+1]], [sigma[i], sigma[i+1]]]) for i in range(len(s)-1)]
    # lc = mc.LineCollection(lines[:int(x.size*0.98)], cmap="jet", lw=2)       # Avoid double-plotting from s by slicing ~ 990
    # # print(np.where(np.diff(s)<0))
    # # Set the values used to determine the color
    # lc.set_array(range(len(s)))
    # ax_collection = ax.add_collection(lc)
    # colorbar = fig.colorbar(ax_collection, ax=ax)

    # ax.plot(s, sigma*s*(s - 4*m_X**2), linestyle='none')        # |M|^2 integrated over t 
    ax.plot(s, sigma, label='vector')
    ax.plot(s, np.vectorize(sigma_pp_dd)(s=s, m_d=m_d, m_phi=m_X, vert=vert), label='scalar')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

    res = C_n_XX_dd(m_d=m_d, m_X=m_X, k_d=-1., k_X=1., T_d=T, xi_d=xi_d, xi_X=xi_X, vert=vert, type=0)
    print(res)
