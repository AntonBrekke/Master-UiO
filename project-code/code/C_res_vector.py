#! /usr/bin/env python3

from scipy.integrate import quad
import numpy as np
import numba as nb
import cmath as cm
from math import exp, log, sqrt, pi, fabs, atan, asin, tan, isfinite
import vegas
from scipy.special import kn
import densities as dens
from scipy.integrate import quad
import vector_mediator

max_exp_arg = 3e2
rtol_int = 1e-4
spin_stat_irr = 1e3
fac_res_width = 1e4
offset = 1.+1e-14

"""
Changes: 
--------
Updated er_C_n_pp_dd_s_t_integral to ker_C_n_XX_dd_s_t_integral (_2)
Updated sigma_pp_dd to sigma_XX_dd
Updated matrix elements 
    * M2_dd 
    * M2_aa 
    * M2_da
which updates C_n_3_12 and C_rho_3_12.
Solved t-integration in ker_C_34_12_s_t_integral
by adding Monte-Carlo integration as analytical result 
was dificult

What needs to be done: 
------------------
Gamma_scat (?, not called anywhere)

------------
-- Comments: 
    To change C_n_3_12, C_rho_3_12, update matrix elements in 
    * pandemolator.py 
    * sterile_caller.py 
    * sterile_pandemic.py 
    * C_res_vector_no_spin_stat.py
    M2_dd = 2.*y2*(c_th**4.)/m_X2 * (m_X2)*(2*m_X2 + (m_d + m_d)**2)
    M2_aa = 2.*y2*(s_th**4.)/m_X2 * (m_X2)*(2*m_X2)
    M2_da = 2.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d**2)*(2*m_X2 + m_d**2)
"""

@nb.jit(nopython=True, cache=True)
def ker_C_n_3_12_E2(log_E2, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type):
    """
    For some reason, matrix element is taken as input. 
    You can find them in 
    * pandemolator.py 
    * sterile_caller.py 
    * sterile_pandemic.py 
    if they need to be changed
    """
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

    res = E2*dist           # Anton: Factor E2 comes from substitution v = ln(E2) in integral
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

    res = E1*res_2          # Anton: Factor E1 comes from substitution u = ln(E1) for integral
    if not isfinite(res):
        return 0.
    return res

# Anton: This is never called ? 
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
def C_n_3_12(m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, M2, type=0):
    """
    Anton: Momentum 3 has been eliminated in this collision operator. 
    p1_vec = p1(0,0,1), p2_vec = p2(sin(th), 0, cos(th))
    """
    E1_min = max(m1, 1e-200)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)    # Infinity, could use np.inf in quad 

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

    res = E2*Etype*dist         # Anton: Factor E2 comes from substitution v = ln(E2) in integral
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

    res = E1*res_2          # Anton: Factor E1 comes from substitution u = ln(E1) in integral
    if not isfinite(res):
        return 0.
    return res

def C_rho_3_12(type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, M2):
    E1_min = max(m1, 1e-10*T1)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)

    res, err = quad(ker_C_rho_3_12_E1, log(E1_min), log(E1_max), args=(type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3), epsabs=0., epsrel=rtol_int)

    return M2*res/(32.*(pi**3.))


# For XX --> dd (for some reason, instead of dd --> XX)
@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s_t_integral(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_X, vert):
    """
    Solved for matrix-element and integrated over variable t in collision
    operator using Mathematica, int_Rt |M|^2/sqrt((t-tm)*(t-tp)). 
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

    # t_max,min - t_m,p = 2p1*p3*(ct_max - ct_m)
    t_add = m_d2 + m_X2
    t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
    t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
    t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)
    
    """
    Anton: 
    x = t - tm, y = t - tp 
    sqrt(x)*sqrt(y) / sqrt(x*y) appears in expressions, makes trouble. 
    This is either 1 for (x>0 or y>0) and -1 for (x<0 and y<0).
    ill-defined for x->0, y->0 wrt. limit to 0+ or 0-.
    x>=0 & y>=0 -> 1
    x>=0 & y<=0 -> 1
    x<=0 & y>=0 -> 1
    x<=0 & y<=0 -> -1
    t_m > t_p always. 
    X_max = t_max - t_m, Y_max = t_max - t_p
    X_min = t_min - t_m, Y_min = t_min - t_p
    For all non-zero integration regions of ct_max, ct_min
    Y_max > 0 always => sqrt_fac_tmax = 1.
    Always have X_min < 0, and Y_min >= 0. For = case, 
    as ct_min --> c_p from above, Y_min --> 0+.
    Therefore, sqrt_fac_tmin = 1 as well.
    """
    # sqrt_fac_tmin = np.ones(n)
    # sqrt_fac_tmax = np.ones(n)
    # X_max = t_max - t_m
    # Y_max = t_max - t_p
    # X_min = t_min - t_m
    # Y_min = t_min - t_p

    # sqrt_fac_tmax[np.logical_and(X_max < 0, Y_max < 0)] = -1
    # sqrt_fac_tmin[np.logical_and(X_min < 0, Y_min < 0)] = -1
    
    sqrt_fac_tmax = 1
    sqrt_fac_tmin = 1

    # Anton: Trick to make numpy evaluate negative numbers in sqrt
    t_min = t_min.astype(np.complex128)
    t_max = t_max.astype(np.complex128)
    t_m = t_m.astype(np.complex128)
    t_p = t_p.astype(np.complex128)
    s = s.astype(np.complex128)
    s2 = s2.astype(np.complex128)
    a = a.astype(np.complex128)

    # Anton: Write each term of t-integrated matrix element sorted by denominators, evaluated at t_min and t_max. 
    term1_max = -((8*(2*m_d2+m_X2)**2*np.sqrt(a*(t_max-t_m)*(t_max-t_p))*(2*m_d6+m_d4*(6*m_X2-3*s-2*(t_max+t_m+t_p))+m_d2*(12*m_X4-4*m_X2*(3*s+t_max+t_m+t_p)+3*s2+2*s*(t_max+t_m+t_p)+2*t_p*(t_max+t_m)+2*t_max*t_m)+(2*m_X2-s)*(2*m_X2-s-t_max)*(2*m_X2-s-t_m)-t_p*(4*m_X4-2*m_X2*(2*s+t_max+t_m)+s2+s*(t_max+t_m)+2*t_max*t_m)))/(a*(t_max-m_d2)*(t_m-m_d2)*(t_p-m_d2)*(-m_d2-2*m_X2+s+t_max)*(-m_d2-2*m_X2+s+t_m)*(-m_d2-2*m_X2+s+t_p)))

    term1_min = -((8*(2*m_d2+m_X2)**2*np.sqrt(a*(t_min-t_m)*(t_min-t_p))*(2*m_d6+m_d4*(6*m_X2-3*s-2*(t_min+t_m+t_p))+m_d2*(12*m_X4-4*m_X2*(3*s+t_min+t_m+t_p)+3*s2+2*s*(t_min+t_m+t_p)+2*t_p*(t_min+t_m)+2*t_min*t_m)+(2*m_X2-s)*(2*m_X2-s-t_min)*(2*m_X2-s-t_m)-t_p*(4*m_X4-2*m_X2*(2*s+t_min+t_m)+s2+s*(t_min+t_m)+2*t_min*t_m)))/(a*(t_min-m_d2)*(t_m-m_d2)*(t_p-m_d2)*(-m_d2-2*m_X2+s+t_min)*(-m_d2-2*m_X2+s+t_m)*(-m_d2-2*m_X2+s+t_p)))

    term2_max = -((4*np.sqrt(a)*sqrt_fac_tmax*(16*m_d8-16*m_d6*(t_m+t_p)-2*m_d4*(12*m_X4+4*m_X4*(-s+t_m+t_p)+s2-2*s*(t_m+t_p)-8*t_m*t_p)+2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p))-2*m_X2*(s*(t_m+t_p)-4*t_m*t_p)+s*(s*(t_m+t_p)-4*t_m*t_p))+2*m_X6*(t_m+t_p)-m_X4*(s*(t_m+t_p)+8*t_m*t_p)-2*s2*t_m*t_p)*(np.log(m_d2-t_max)-np.log(m_d2*(2*t_max-t_m-t_p)+2*np.sqrt(m_d2-t_m)*np.sqrt(m_d2-t_p)*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)-t_max*(t_m+t_p)+2*t_m*t_p)))/(a*(m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)*(2*m_X2-s)))

    term2_min = -((4*np.sqrt(a)*sqrt_fac_tmin*(16*m_d8-16*m_d6*(t_m+t_p)-2*m_d4*(12*m_X4+4*m_X4*(-s+t_m+t_p)+s2-2*s*(t_m+t_p)-8*t_m*t_p)+2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p))-2*m_X2*(s*(t_m+t_p)-4*t_m*t_p)+s*(s*(t_m+t_p)-4*t_m*t_p))+2*m_X6*(t_m+t_p)-m_X4*(s*(t_m+t_p)+8*t_m*t_p)-2*s2*t_m*t_p)*(np.log(m_d2-t_min)-np.log(m_d2*(2*t_min-t_m-t_p)+2*np.sqrt(m_d2-t_m)*np.sqrt(m_d2-t_p)*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)-t_min*(t_m+t_p)+2*t_m*t_p)))/(a*(m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)*(2*m_X2-s)))

    term3_max = (4*np.sqrt(a)*sqrt_fac_tmax*(16*m_d8+16*m_d6*(6*m_X2-3*s-t_m-t_p)+2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p)+8*t_m*t_p)+2*m_d2*(34*m_X6-m_X4*(57*s+16*(t_m+t_p))+2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)-s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p))-24*m_X8+2*m_X6*(12*s+7*(t_m+t_p))-m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)+4*m_X2*s2*(2*s+t_m+t_p)-2*s2*(s+t_m)*(s+t_p))*(np.log(m_d2+2*m_X2-s-t_max)-np.log(2*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)*np.sqrt(m_d2+2*m_X2-s-t_m)*np.sqrt(m_d2+2*m_X2-s-t_p)+m_d2*(2*t_max-t_m-t_p)+m_X2*(4*t_max-2*(t_m+t_p))+t_p*(s-t_max+2*t_m)-2*s*t_max+s*t_m-t_max*t_m)))/(a*(2*m_X2-s)*(m_d2+2*m_X2-s-t_m)**(3/2)*(m_d2+2*m_X2-s-t_p)**(3/2))

    term3_min = (4*np.sqrt(a)*sqrt_fac_tmin*(16*m_d8+16*m_d6*(6*m_X2-3*s-t_m-t_p)+2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p)+8*t_m*t_p)+2*m_d2*(34*m_X6-m_X4*(57*s+16*(t_m+t_p))+2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)-s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p))-24*m_X8+2*m_X6*(12*s+7*(t_m+t_p))-m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)+4*m_X2*s2*(2*s+t_m+t_p)-2*s2*(s+t_m)*(s+t_p))*(np.log(m_d2+2*m_X2-s-t_min)-np.log(2*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)*np.sqrt(m_d2+2*m_X2-s-t_m)*np.sqrt(m_d2+2*m_X2-s-t_p)+m_d2*(2*t_min-t_m-t_p)+m_X2*(4*t_min-2*(t_m+t_p))+t_p*(s-t_min+2*t_m)-2*s*t_min+s*t_m-t_min*t_m)))/(a*(2*m_X2-s)*(m_d2+2*m_X2-s-t_m)**(3/2)*(m_d2+2*m_X2-s-t_p)**(3/2))

    term4_max = -((16*sqrt_fac_tmax*np.log(2*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)+2*t_max-t_m-t_p))/np.sqrt(a))
    term4_min = -((16*sqrt_fac_tmin*np.log(2*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)+2*t_min-t_m-t_p))/np.sqrt(a))

    int_tmax = term1_max + term2_max + term3_max + term4_max
    int_tmin = term1_min + term2_min + term3_min + term4_min

    # print(np.max(np.abs(int_tmax - int_tmin).imag))     # Should be 0.0

    return vert*(int_tmax - int_tmin).real


@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s(s, E1, E2, E3, p1, p3, m_d, m_X, s12_min, s12_max, s34_min, s34_max, vert):
    p12 = p1*p1
    p32 = p3*p3
    # Anton: a,b,c definition in a*cos^2 + b*cos + c = 0 in integrand
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2))*(s-2.*E3*(E1+E2))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    # Anton: ct_p, ct_m solutions of a*cos^2 + b*cos + c = 0 for cos. ct_m > ct_p
    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)
    # Anton: R_theta integration region {-1 <= cos <= 1 | c_p <= cos <= c_m}
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    in_res = (ct_max > ct_min)

    # Anton: return zero for integral if it is not inside defined region.
    # x = [0,0,0], x[[True,False,True]] = [1,2,3] => x = [1,0,3]
    t_int = np.zeros(s.size)
    t_int[in_res] = ker_C_n_XX_dd_s_t_integral(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, vert)
    return t_int

# 3 4 -> 1 2 <=> X X -> d d
@nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd(x, m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, vert):
    """
    Anton: Seems like E1 <--> E3, E2 <--> E4 compared to article.
    Set up for production of X. 
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
    f3 = exp_3/(1. + k_X*exp_3)
    f4 = exp_4/(1. + k_X*exp_4)
    # Anton: For X initial, d final
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
def C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, vert, type=0):
    """
    Anton: 
    Collision operator C[X]_XX_dd for X, C[X]_XX_dd = -C[d]_XX_dd.
    1,2 = d,d, 3,4 = X,X
    
    dist = f3*f4*f1t*f2t
    type = -1, XX --> dd
    -f3*f4*f1t*f2t = -1*dist
    --> chem_eq_fac = -1

    type = 0, XX <--> dd
    f1*f2*f3t*f4t - f1t*f2t*f3*f4 = (exp(xi1 + xi2 - xi3 - x4) - 1)*dist
    --> chem_eq_fac = (exp(2*(xi_d - xi_X)) - 1)

    type = 1, dd --> XX
    f1*f2*f3t*f4t = exp(xi1 + xi - xi3 - xi4)*dist
    --> chem_eq_fac = exp(2*(xi_d - xi_X))

    and same procedure in the spin_stat_irr case. 
    """
    if m_X/T_d - xi_X > spin_stat_irr: # spin-statistics irrelevant here
        th_avg_s_v = th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert)
        if th_avg_s_v <= 0.:
            if type == 2:
                return np.array([0., 0.])
            return 0.
        if type == 0:
            chem_eq_fac = exp(2.*xi_d) - exp(2.*xi_X)
        elif type == -1:
            chem_eq_fac = -exp(2.*xi_X)
        elif type == 1:
            chem_eq_fac = exp(2.*xi_d)
        elif type == 2:
            return np.array([-exp(2.*xi_X), exp(2.*xi_d)])*th_avg_s_v
        return chem_eq_fac*th_avg_s_v

    # Anton: Factors to get correct 'dist' in ker_C_n_XX_dd for different types
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
def M2_XX_dd(s, t, m_d2, vert, m_X2):
    m_d4 = m_d2*m_d2
    m_d8 = m_d4*m_d4
    m_X4 = m_X2*m_X2

    t2 = t*t
    t3 = t*t2
    u = 2*m_d2 + 2*m_X2 - s - t
    u2 = u*u
    u3 = u*u2

    M2 = -(8*(6*m_d8-m_d4*(6*m_X4-4*m_X2*(t+u)+3*t2+14*t*u+3*u2)+m_d2*(6*m_X4*(t+u)-16*m_X2*t*u+t3+7*t2*u+7*t*u2+u3)+m_X4*(t2-8*t*u+u2)+4*m_X2*t*u*(t+u)-t*u*(t2+u2)))/((m_d2-t)**2*(m_d2-u)**2)

    return vert*M2 

# @nb.jit(nopython=True, cache=True)
def ker_C_34_12_s_t_integral(s, t, a, ct_m, ct_p, E1, E3, p1, p3, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, res_sub=False):
    m12 = m1*m1
    m32 = m3*m3
    t_add = m12 + m32
    t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)

    integrand = vector_mediator.M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, res_sub) * 1/np.sqrt(a*(t - t_m)*(t - t_p))
    return integrand

# @nb.jit(nopython=True, cache=True)
def ker_C_34_12_s(s, E1, E2, E3, ct, p1, p2, p3, ct_min, ct_max, ct_m, ct_p, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, res_sub):
    # t = E1**2 + E3**2 - p1**2 + p3**2 + 2*p1*p3*ct
    t = (E1-p1)*(E1+p1) + (E3-p3)*(E3+p3) + 2*p1*p3*ct
    in_res = (ct_max > ct_min)

    t_int = np.zeros(s.size)

    # Anton: New
    t_int[in_res] = ker_C_34_12_s_t_integral(s[in_res], t[in_res], a[in_res], ct_m[in_res], ct_p[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m1, m2, m3, m4, vert, m_X2, m_Gamma_X2[in_res], res_sub)

    return t_int

# @nb.jit(nopython=True, cache=True)
def ker_C_34_12(x, log_s_min, log_s_max, type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_X2, m_Gamma_X2, res_sub, thermal_width):
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    m42 = m4*m4

    # Anton: Integration order is switched, now it is s, E1, E2, E3, t
    s = np.exp(np.fmin(log_s_min * (1.-x[:,0]) + log_s_max * x[:,0], 6e2))

    E1_min = m1
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)
    log_E1_min = log(E1_min)
    log_E1_max = log(E1_max)
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

    E4 = E1 + E2 - E3
    p4 = np.sqrt(np.fmax((E4-m4)*(E4+m4), 1e-200))

    s12_min = m12 + m22 + 2.*E1*(E2 - p1*p2/E1)
    s12_max = m12 + m22 + 2.*E1*(E2 + p1*p2/E1)
    s34_min = m32 + m42 + 2.*E3*(E4 - p3*p4/E3)
    s34_max = m32 + m42 + 2.*E3*(E4 + p3*p4/E3)

    p12 = p1*p1
    p32 = p3*p3
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2)+(m1-m2)*(m1+m2))*(s-2.*E3*(E1+E2)+(m3-m4)*(m3+m4))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))
    ct_p = (-b+sqrt_fac)/(2.*a)
    ct_m = (-b-sqrt_fac)/(2.*a)
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    
    ct = x[:,4]

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
        m_X = sqrt(m_X2)
        sqrt_arg = (m_X2-4.*m3*m3)*((E1+E2)**2.-m_X2)
        sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 1e-200))
        E3p = 0.5*(E12+sqrt_fac/m_X)
        E3m = 0.5*(E12-sqrt_fac/m_X)
        exp_3p_xi = np.exp(np.fmin(xi3-E3p/T3, max_exp_arg))
        exp_3m_xi = np.exp(np.fmin(xi3-E3m/T3, max_exp_arg))
        E3_integral = sqrt_fac/(T3*m_X) + np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))
        m_Gamma_X_T = sqrt(m_Gamma_X2)*(1.+m_X*T3*np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))/sqrt_fac)
        m_Gamma_X_T2 = m_Gamma_X_T*m_Gamma_X_T
    else:
        m_Gamma_X_T2 = m_Gamma_X2*np.ones(s.size)

    ker_s = ker_C_34_12_s(s, E1, E2, E3, ct, p1, p2, p3, ct_min, ct_max, ct_m, ct_p, m1, m2, m3, m4, vert, m_X2, m_Gamma_X_T2, res_sub)

    jac = E1*(log_E1_max-log_E1_min)*E2*(log_E2_max-log_E2_min)*E3*(log_E3_max-log_E3_min)*s*(log_s_max-log_s_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    return res

# 3 4 -> 1 2 (all neutrinos); nFW (nBW): # of particle occurence in final-initial state for forward 3 4 -> 1 2 (backward 1 2 -> 3 4) reaction
# type indicates if for n (0) or rho (1 for E1, 2 for E2, 3 for E3, 4 for E4, 12 for E1+E2 = E3+E4)
# note that when using thermal width it is assumed that m3 = m4 = md, T3 = T4 = Td, xi3 = xi4 = xid, xi_phi = 2 xi_d
# and 3, 4 are fermions, phi is boson
# Anton: Added Monte-Carlo integral for t-integration
def C_34_12(type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_X2, m_Gamma_X2, res_sub=False, thermal_width=True):
    # Anton: Integration order is now s, E1, E2, E3, t
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
    s_vals = np.sort(np.array([s_min, s_max, m_X2-fac_res_width*sqrt(m_Gamma_X2), m_X2, m_X2+fac_res_width*sqrt(m_Gamma_X2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    np.seterr(divide='ignore')
    for i in range(len(s_vals)-1):
        @vegas.batchintegrand
        def kernel(x):
            return ker_C_34_12(x, log(s_vals[i]), log(s_vals[i+1]), type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_X2, m_Gamma_X2, res_sub, thermal_width)
        integ = vegas.Integrator(4 * [[0., 1.]] + [[-1, 1]])
        result = integ(kernel, nitn=10, neval=1e5)
        # print(result.summary())
        # if result.mean != 0.:
        #     print("Vegas error 34 12: ", result.sdev/fabs(result.mean), result.mean/(256.*(pi**6.)), result.Q)
        res += result.mean
    np.seterr(divide='warn')
    # print("34 12:", res/(256.*(pi**6.)), (th_avg_sigma_v_33_11(m3, m4, m1, T1, vert, m_phi2, m_Gamma_phi2)*(nBW*exp(xi1+xi2))))

    return res/(256.*(pi**6.))

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
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3 + m2/4)^2]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space H(E_cm - m3 - m4), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1 + m2)]^2 - 4*m1^2*m2^2}) = H([s - (m1 + m2)^2]^2 - 4*m1^2*m2^2)
    = H(s - m1 - m2 - 2*m1*m2) = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*p1cm^2) 
          * int_{t_lower}^{t_upper} dt |M|^2
    Note: This function can be vectorized, but is not needed. 
          Use np.vectorize(sigma_XX_dd)(s, m_d, m_X, vert) instead if array output is wanted.
    """
    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    m_d4 = m_d2*m_d2
    m_X4 = m_X2*m_X2

    # Anton: Heavyside-functions
    if s < 4*m_d**2 or s < 4*m_X**2:
        return 0. 

    s2 = s*s

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_d2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    # Anton: t-integrated squared matrix elements
    # Anton: imaginary parts from upper - lower will cancel
    int_t_M2_upper = 8*vert*((2*m_d2+m_X2)**2/(-m_d2-2*m_X2+s+t_upper) + (2*m_d2+m_X2)**2/(t_upper-m_d2) + ((-8*m_d4+4*m_d2*(s-2*m_X2)+4*m_X4+s2)*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper)))/(2*m_X2-s) - 2*t_upper)

    int_t_M2_lower = 8*vert*((2*m_d2+m_X2)**2/(-m_d2-2*m_X2+s+t_lower) + (2*m_d2+m_X2)**2/(t_lower-m_d2) + ((-8*m_d4+4*m_d2*(s-2*m_X2)+4*m_X4+s2)*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower)))/(2*m_X2-s) - 2*t_lower)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma / 2

# Thermal average 

def ker_th_avg_sigma_v_XX_dd(log_s, T_d, m_d, m_X, vert):
    s = exp(log_s)
    sqrt_s = sqrt(s)
    sigma = sigma_XX_dd(s, m_d, m_X, vert)
    return s*sigma*(s-4.*m_X*m_X)*sqrt_s*kn(1, sqrt_s/T_d)

# only \int d^3 p3 d^3 p4 sigma v exp(-(E3+E4)/T)/(2 pi)^6
def th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert):
    s_min = max(4.*m_d*m_d, 4.*m_X*m_X)
    s_max = (5e2*T_d)**2.
    if s_max <= s_min:
        return 0.

    res, err = quad(ker_th_avg_sigma_v_XX_dd, log(s_min), log(s_max), args=(T_d, m_d, m_X, vert), epsabs=0., epsrel=rtol_int)

    return res*T_d/(32.*(pi**4.))
    # return res/(8.*(m_phi**4.)*T_d*(kn(2, m_phi/T_d)**2.))

def ker_th_avg_sigma_v_33_11(log_s, m1, m2, m3, T, vert, m_X2, m_Gamma_X2, res_sub):
    s = exp(log_s)
    sqrt_s = sqrt(s)
    sigma = vector_mediator.sigma_gen(s, m3, m3, m1, m2, vert, m_X2, m_Gamma_X2, sub=res_sub)
    # print(log_s, s*sigma*(s-4.*m1*m1)*sqrt_s*kn(1, sqrt_s/T3))
    return s*sigma*(s-4.*m3*m3)*sqrt_s*kn(1, sqrt_s/T)

# only \int d^3 p3 d^3 p4 sigma v exp(-(E3+E4)/T)/(2 pi)^6
def th_avg_sigma_v_33_11(m1, m2, m3, T, vert, m_X2, m_Gamma_X2, res_sub=True):
    s_min = max((m1+m2)*(m1+m2), 4.*m3*m3)*offset
    s_max = (1e3*T)**2.
    if s_max <= s_min:
        return 0.
    s_vals = np.sort(np.array([s_min, s_max, m_X2-fac_res_width*sqrt(m_Gamma_X2), m_X2, m_X2+fac_res_width*sqrt(m_Gamma_X2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    for i in range(len(s_vals)-1):
        cur_res, err = quad(ker_th_avg_sigma_v_33_11, log(s_vals[i]), log(s_vals[i+1]), args=(m1, m2, m3, T, vert, m_X2, m_Gamma_X2, res_sub), epsabs=0., epsrel=rtol_int, limit=100)
        res += cur_res

    return res*T/(32.*(pi**4.))

########################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    m_d = 2e-5      # 1e-5 GeV = 10 kev 
    m_a = 0.
    m_X = 3*m_d
    sin2_2th = 1e-15
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    y = 2e-3

    vert = y**4 * np.cos(th)**8
    # vert = 1

    T = 0.6510550394714374
    # xi = mu / T
    xi_d = -8.551301127056323
    xi_X = 0.
    xi_a = 0.

    # Fermion: 1, Boson: -1
    k_d = 1.
    k_a = 1.
    k_X = -1.
    T_d = T
    T_a = T

    # Anton: fi = aa->dd, tr = ad->dd, el = dd->dd
    # Anton: For small theta << 1, sin^2(theta) = 1/4*sin^2(2*theta)
    vert_fi = y**4 * np.cos(th)**4*np.sin(th)**4
    vert_tr = y**4 * np.cos(th)**6*np.sin(th)**2
    vert_el = y**4 * np.cos(th)**8

    ########################################################################
    # Mostly plot things to check 
    x = np.linspace(0, 1, int(1e3))    # x = ln(s/s_min) / ln(s_max/s_min)
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
    # fig = plt.figure()
    # ax = fig.add_subplot()
    E3 = E3_[int((x.size-1)*0.9)]
    E4 = E4_[int((x.size-1)*0.5)]
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
    m_Gamma_X2 = vector_mediator.Gamma_X(y, th, m_X, m_d)**2
    time2 = time.time()
    print(f'vector_mediator.Gamma_X ran in {time.time()-time1}s')
    m1 = m_d
    k1 = k_d
    T1 = T_d
    xi1 = xi_d

    m2 = m_a
    k2 = k_a
    T2 = T_a
    xi2 = xi_a

    m3 = m_d
    k3 = k_d
    T3 = T_d
    xi3 = xi_d

    m4 = m_d
    k4 = k_d
    T4 = T_d
    xi4 = xi_d

    m_X2 = m_X*m_X
    vert_da = y**4 * np.cos(th)**6*np.sin(th)**2
    vert = vert_da

    # fig = plt.figure()
    # ax = fig.add_subplot()

    # t1 = time.time()
    # C_34_12_new_val = C_34_12_new(type=0, nFW=1., nBW=-1., m1=m1, m2=m2, m3=m3, m4=m4, k1=k1, k2=k2, k3=k3, k4=k4, T1=T1, T2=T2, T3=T3, T4=T4, xi1=xi1, xi2=xi2, xi3=xi3, xi4=xi4, vert=vert, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2)
    # print(C_34_12_new_val, 'C_34_12_new_val', time.time()-t1)
    # t1 = time.time()
    # C_34_12_val = C_34_12(type=0, nFW=1., nBW=-1., m1=m1, m2=m2, m3=m3, m4=m4, k1=k1, k2=k2, k3=k3, k4=k4, T1=T1, T2=T2, T3=T3, T4=T4, xi1=xi1, xi2=xi2, xi3=xi3, xi4=xi4, vert=vert, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2)
    # print(C_34_12_val, 'C_34_12_val', time.time()-t1)

    # b = ker_C_34_12_s(s, E1, E2, E3, p1, p2, p3, s12_min, s12_max, s34_min, s34_max, m1, m2, m3, m4, vert, m_X**2, m_Gamma_X2, res_sub=False)
    # time1 = time.time()
    # ker_C_34_12_s_t_integral_val_2 = ker_C_34_12_s_t_integral_2(ct_min=ct_min[in_res], ct_max=ct_max[in_res], ct_p=ct_p[in_res], ct_m=ct_m[in_res], s=s[in_res], E1=E1, E3=E3, p1=p1, p3=p3, m1=m_d, m2=m2, m3=m3, m4=m4, vert=vert, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    # print(f'ker_C_n_12_34_s_t_integral_2 ran in {time.time()-time1}s')

    # ax.plot(s[in_res], ker_C_34_12_s_t_integral_val_2)
    # fig.tight_layout()
    # plt.show()

    from C_res_scalar import C_n_pp_dd

    # N = int(1e1)
    # T_d_arr = 10**np.linspace(-3, np.log10(1.3), N)
    # xi_d_arr = np.linspace(-2e1, 2e8, N)
    # xi_X_arr = np.linspace(-4e1, 4e8, N)
    # C_pp_dd_arr = np.array([C_n_pp_dd(m_d=m_d, m_phi=m_X, k_d=k_d, k_phi=k_X, T_d=T_di, xi_d=xi_di, xi_phi=xi_Xi, vert=vert_el, type=0) for T_di, xi_di, xi_Xi in zip(T_d_arr, xi_d_arr, xi_X_arr)])
    # C_XX_dd_arr = np.array([C_n_XX_dd(m_d=m_d, m_X=m_X, k_d=k_d, k_X=k_X, T_d=T_di, xi_d=xi_di, xi_X=xi_Xi, vert=vert_el, type=0) for T_di, xi_di, xi_Xi in zip(T_d_arr, xi_d_arr, xi_X_arr)])
    # plt.plot(m_d/T_d_arr, abs(C_pp_dd_arr))
    # plt.plot(m_d/T_d_arr, abs(C_XX_dd_arr))
    # plt.show()

    # Anton: Want to compare XX to dd 
    @nb.jit(nopython=True, cache=True)
    def ker_C_n_pp_dd_s_t_integral(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_phi, vert):
        """
        Solved for matrix-element and integrated over variable t in collision
        operator using Mathematica, int_Rt |M|^2/sqrt((t-tm)*(t-tp)). 
        These expressions can be found in nu_s_nu_s_to_XX.nb
        """
        n = s.size
        s2 = s*s
        m_d2 = m_d*m_d
        m_d4 = m_d2*m_d2
        m_d6 = m_d2*m_d4
        m_d8 = m_d4*m_d4
        m_phi2 = m_phi*m_phi
        m_phi4 = m_phi2*m_phi2
        m_phi6 = m_phi2*m_phi4
        m_phi8 = m_phi4*m_phi4

        t_add = m_d2 + m_X2
        t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
        t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
        t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
        t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)
        
        # Anton: Write each term of t-integrated matrix element sorted by denominators, evaluated at t_min and t_max. 
        """
        sqrt(x)*sqrt(y) / sqrt(x*y) makes trouble. 
        This is either 1 for (x>=0, y>=0), (x>=0, y<0), (x<0, y>=0) and -1 for (x<0, y<0).
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
        
        # Anton: Trick to make numpy evaluate negative numbers 
        t_min = t_min + 0j
        t_max = t_max + 0j
        t_m = t_m + 0j
        t_p = t_p + 0j
        s = s + 0j
        s2 = s2 + 0j
        
        a = -a

        # Anton: When t_max = t_m, t_min = t_p, max - min always cancel ... 
        term1_max = -((2*(m_phi2-4*m_d2)**2*np.sqrt(a*(t_max-t_m)*(t_max-t_p))*(8*m_phi6+4*m_phi4*(3*m_d2-3*s-t_max-t_m-t_p)+2*m_phi2*(3*m_d4-2*m_d2*(3*s+t_max+t_m+t_p)+3*s2+2*s*(t_max+t_m+t_p)+t_p*(t_max+t_m)+t_max*t_m)+2*m_d6-m_d4*(3*s+2*(t_max+t_m+t_p))+m_d2*(3*s2+2*s*(t_max+t_m+t_p)+2*t_max*(t_m+t_p)+2*t_m*t_p)-t_p*(s2+s*(t_max+t_m)+2*t_max*t_m)-s*(s+t_max)*(s+t_m)))/(a*(t_max-m_d2)*(t_m-m_d2)*(t_p-m_d2)*(-2*m_phi2-m_d2+s+t_max)*(-2*m_phi2-m_d2+s+t_m)*(-2*m_phi2-m_d2+s+t_p)))

        term1_min = -((2*(m_phi2-4*m_d2)**2*np.sqrt(a*(t_min-t_m)*(t_min-t_p))*(8*m_phi6+4*m_phi4*(3*m_d2-3*s-t_min-t_m-t_p)+2*m_phi2*(3*m_d4-2*m_d2*(3*s+t_min+t_m+t_p)+3*s2+2*s*(t_min+t_m+t_p)+t_p*(t_min+t_m)+t_min*t_m)+2*m_d6-m_d4*(3*s+2*(t_min+t_m+t_p))+m_d2*(3*s2+2*s*(t_min+t_m+t_p)+2*t_min*(t_m+t_p)+2*t_m*t_p)-t_p*(s2+s*(t_min+t_m)+2*t_min*t_m)-s*(s+t_min)*(s+t_m)))/(a*(t_min-m_d2)*(t_m-m_d2)*(t_p-m_d2)*(-2*m_phi2-m_d2+s+t_min)*(-2*m_phi2-m_d2+s+t_m)*(-2*m_phi2-m_d2+s+t_p)))

        term2_max = (1/(a*(2*m_phi2-s)*(m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)))*np.sqrt(a)*sqrt_fac_tmax*(m_phi6*(4*m_d2-2*(t_m+t_p))+m_phi4*(-20*m_d4+m_d2*(4*(t_m+t_p)-2*s)+s*(t_m+t_p)+12*t_m*t_p)+8*m_phi2*(4*m_d2+s)*(m_d4-t_m*t_p)-64*m_d8+64*m_d6*(t_m+t_p)+2*m_d4*(s2-8*s*(t_m+t_p)-32*t_m*t_p)-2*m_d2*s*(s*(t_m+t_p)-16*t_m*t_p)+2*s2*t_m*t_p)*(np.log(m_d2-t_max)-np.log(m_d2*(2*t_max-t_m-t_p)+2*np.sqrt(m_d2-t_m)*np.sqrt(m_d2-t_p)*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)-t_max*(t_m+t_p)+2*t_m*t_p))

        term2_min = (1/(a*(2*m_phi2-s)*(m_d2-t_m)**(3/2)*(m_d2-t_p)**(3/2)))*np.sqrt(a)*sqrt_fac_tmin*(m_phi6*(4*m_d2-2*(t_m+t_p))+m_phi4*(-20*m_d4+m_d2*(4*(t_m+t_p)-2*s)+s*(t_m+t_p)+12*t_m*t_p)+8*m_phi2*(4*m_d2+s)*(m_d4-t_m*t_p)-64*m_d8+64*m_d6*(t_m+t_p)+2*m_d4*(s2-8*s*(t_m+t_p)-32*t_m*t_p)-2*m_d2*s*(s*(t_m+t_p)-16*t_m*t_p)+2*s2*t_m*t_p)*(np.log(m_d2-t_min)-np.log(m_d2*(2*t_min-t_m-t_p)+2*np.sqrt(m_d2-t_m)*np.sqrt(m_d2-t_p)*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)-t_min*(t_m+t_p)+2*t_m*t_p))

        term3_max = -((np.sqrt(a)*sqrt_fac_tmax*(40*m_phi8-2*m_phi6*(10*m_d2+36*s+11*(t_m+t_p))+m_phi4*(-468*m_d4+6*m_d2*(23*s+6*(t_m+t_p))+50*s2+27*s*(t_m+t_p)+12*t_m*t_p)-4*m_phi2*(88*m_d6-6*m_d4*(23*s+8*(t_m+t_p))+m_d2*(30*s2+20*s*(t_m+t_p)+8*t_m*t_p)+s*(4*s2+3*s*(t_m+t_p)+2*t_m*t_p))+2*(-32*m_d8+32*m_d6*(3*s+t_m+t_p)-m_d4*(79*s2+56*s*(t_m+t_p)+32*t_m*t_p)+m_d2*s*(14*s2+15*s*(t_m+t_p)+16*t_m*t_p)+s2*(s+t_m)*(s+t_p)))*(np.log(2*m_phi2+m_d2-s-t_max)-np.log(2*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)*np.sqrt(2*m_phi2+m_d2-s-t_m)*np.sqrt(2*m_phi2+m_d2-s-t_p)+m_phi2*(4*t_max-2*(t_m+t_p))+m_d2*(2*t_max-t_m-t_p)+t_p*(s-t_max+2*t_m)-2*s*t_max+s*t_m-t_max*t_m)))/(a*(2*m_phi2-s)*(2*m_phi2+m_d2-s-t_m)**(3/2)*(2*m_phi2+m_d2-s-t_p)**(3/2)))

        term3_min = -((np.sqrt(a)*sqrt_fac_tmin*(40*m_phi8-2*m_phi6*(10*m_d2+36*s+11*(t_m+t_p))+m_phi4*(-468*m_d4+6*m_d2*(23*s+6*(t_m+t_p))+50*s2+27*s*(t_m+t_p)+12*t_m*t_p)-4*m_phi2*(88*m_d6-6*m_d4*(23*s+8*(t_m+t_p))+m_d2*(30*s2+20*s*(t_m+t_p)+8*t_m*t_p)+s*(4*s2+3*s*(t_m+t_p)+2*t_m*t_p))+2*(-32*m_d8+32*m_d6*(3*s+t_m+t_p)-m_d4*(79*s2+56*s*(t_m+t_p)+32*t_m*t_p)+m_d2*s*(14*s2+15*s*(t_m+t_p)+16*t_m*t_p)+s2*(s+t_m)*(s+t_p)))*(np.log(2*m_phi2+m_d2-s-t_min)-np.log(2*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)*np.sqrt(2*m_phi2+m_d2-s-t_m)*np.sqrt(2*m_phi2+m_d2-s-t_p)+m_phi2*(4*t_min-2*(t_m+t_p))+m_d2*(2*t_min-t_m-t_p)+t_p*(s-t_min+2*t_m)-2*s*t_min+s*t_m-t_min*t_m)))/(a*(2*m_phi2-s)*(2*m_phi2+m_d2-s-t_m)**(3/2)*(2*m_phi2+m_d2-s-t_p)**(3/2)))

        term4_max = -(8*sqrt_fac_tmax*np.log(2*np.sqrt(t_max-t_m)*np.sqrt(t_max-t_p)+2*t_max-t_m-t_p))/np.sqrt(a)

        term4_min = -(8*sqrt_fac_tmin*np.log(2*np.sqrt(t_min-t_m)*np.sqrt(t_min-t_p)+2*t_min-t_m-t_p))/np.sqrt(a)

        int_max = term1_max + term2_max + term3_max + term4_max
        int_min = term1_min + term2_min + term3_min + term4_min

        # print(np.max(np.abs(int_max - int_min).imag))     # Should be 0.0

        # Term2 makes noise 
        return vert*(int_max - int_min).real


    # Anton: Also want to compare with Depta
    # @nb.jit(nopython=True, cache=True)
    def ker_C_n_pp_dd_s_t_integral_Depta(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_phi, vert):
        """
        Solved for matrix-element and integrated over variable t in collision
        operator using Mathematica, int_Rt |M|^2/sqrt((t-tm)*(t-tp)). 
        These expressions can be found in nu_s_nu_s_to_XX.nb
        """
        n = s.size
        s2 = s*s
        m_d2 = m_d*m_d
        m_d4 = m_d2*m_d2
        m_d6 = m_d2*m_d4
        m_d8 = m_d4*m_d4
        m_phi2 = m_phi*m_phi
        m_phi4 = m_phi2*m_phi2
        m_phi6 = m_phi2*m_phi4
        m_phi8 = m_phi4*m_phi4

        t_add = m_d2 + m_X2
        t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
        t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
        t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
        t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)

        # Anton: Trick to make numpy evaluate negative numbers 
        t_min = t_min + 0j
        t_max = t_max + 0j
        t_m = t_m + 0j
        t_p = t_p + 0j
        s = s + 0j
        s2 = s2 + 0j
        
        # Anton: Write each term of t-integrated matrix element sorted by denominators, evaluated at t_min and t_max. 
        """
        sqrt(x)*sqrt(y) / sqrt(x*y) makes trouble. 
        This is either 1 for (x>=0, y>=0), (x>=0, y<0), (x<0, y>=0) and -1 for (x<0, y<0).
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

        atan_fac_tmin = np.zeros(n)
        atan_fac_tmax = np.zeros(n)

        atan_fac_tmax[np.where(t_max==t_p)] = np.pi/2
        atan_fac_tmax[np.where(t_max==t_m)] = 0
        neq_max = np.logical_and(t_max!=t_p, t_max!=t_m)
        atan_fac_tmax[neq_max] = np.arctan((np.sqrt(m_d2-t_p[neq_max])*np.sqrt(t_max[neq_max]-t_m[neq_max]))/(np.sqrt(t_m[neq_max]-m_d2)*np.sqrt(t_max[neq_max]-t_p[neq_max])))

        atan_fac_tmin[np.where(t_min==t_p)] = np.pi/2
        atan_fac_tmin[np.where(t_min==t_m)] = 0
        neq_min = np.logical_and(t_min!=t_p, t_min!=t_m)
        atan_fac_tmin[neq_min] = np.arctan((np.sqrt(m_d2-t_p[neq_min])*np.sqrt(t_min[neq_min]-t_m[neq_min]))/(np.sqrt(t_m[neq_min]-m_d2)*np.sqrt(t_min[neq_min]-t_p[neq_min])))

        # Anton: When t_max = t_m, t_min = t_p, max - min always cancel ... 
        term1_max = -2*m_phi4*np.sqrt((t_max-t_m)*(t_max-t_p))/((m_d2-t_max)*(m_d2-t_m)*(m_d2-t_p))

        term1_min = -2*m_phi4*np.sqrt((t_min-t_m)*(t_min-t_p))/((m_d2-t_min)*(m_d2-t_m)*(m_d2-t_p))

        term2_max = -((2*m_phi4*np.sqrt((t_max-t_m)*(t_max-t_p)))/((2*m_phi2+m_d2-s-t_max)*(2*m_phi2+m_d2-s-t_m)*(2*m_phi2+m_d2-s-t_p)))

        term2_min = -((2*m_phi4*np.sqrt((t_min-t_m)*(t_min-t_p)))/((2*m_phi2+m_d2-s-t_min)*(2*m_phi2+m_d2-s-t_m)*(2*m_phi2+m_d2-s-t_p)))

        term3_max = (2*sqrt_fac_tmax*(m_phi6*(4*m_d2-2*(t_m+t_p))+m_phi4*(12*m_d4-2*m_d2*(s+6*(t_m+t_p))+s*(t_m+t_p)+12*t_m*t_p)-8*m_phi2*s*(m_d2-t_m)*(m_d2-t_p)+2*s2*(m_d2-t_m)*(m_d2-t_p))*atan_fac_tmax)/((s-2*m_phi2)*(t_m-m_d2)**(3/2)*(m_d2-t_p)**(3/2))

        term3_min = (2*sqrt_fac_tmin*(m_phi6*(4*m_d2-2*(t_m+t_p))+m_phi4*(12*m_d4-2*m_d2*(s+6*(t_m+t_p))+s*(t_m+t_p)+12*t_m*t_p)-8*m_phi2*s*(m_d2-t_m)*(m_d2-t_p)+2*s2*(m_d2-t_m)*(m_d2-t_p))*atan_fac_tmin)/((s-2*m_phi2)*(t_m-m_d2)**(3/2)*(m_d2-t_p)**(3/2))

        term4_max = -((2*sqrt_fac_tmax*(40*m_phi8+m_phi6*(44*m_d2-72*s-22*(t_m+t_p))+m_phi4*(12*m_d4-6*m_d2*(9*s+2*(t_m+t_p))+50*s2+27*s*(t_m+t_p)+12*t_m*t_p)-4*m_phi2*s*(2*m_d4-2*m_d2*(3*s+t_m+t_p)+4*s2+3*s*(t_m+t_p)+2*t_m*t_p)+2*s2*(-m_d2+s+t_m)*(-m_d2+s+t_p))*atan_fac_tmax)/((s-2*m_phi2)*(-2*m_phi2-m_d2+s+t_m)**(3/2)*(2*m_phi2+m_d2-s-t_p)**(3/2)))

        term4_min = -((2*sqrt_fac_tmin*(40*m_phi8+m_phi6*(44*m_d2-72*s-22*(t_m+t_p))+m_phi4*(12*m_d4-6*m_d2*(9*s+2*(t_m+t_p))+50*s2+27*s*(t_m+t_p)+12*t_m*t_p)-4*m_phi2*s*(2*m_d4-2*m_d2*(3*s+t_m+t_p)+4*s2+3*s*(t_m+t_p)+2*t_m*t_p)+2*s2*(-m_d2+s+t_m)*(-m_d2+s+t_p))*atan_fac_tmin)/((s-2*m_phi2)*(-2*m_phi2-m_d2+s+t_m)**(3/2)*(2*m_phi2+m_d2-s-t_p)**(3/2)))

        term5_max = -16*sqrt_fac_tmax*np.log(np.sqrt(t_max-t_m)+np.sqrt(t_max-t_p))
        term5_min = -16*sqrt_fac_tmin*np.log(np.sqrt(t_min-t_m)+np.sqrt(t_min-t_p))

        return vert*(term1_max + term2_max + term3_max + term4_max + term5_max - term1_min - term2_min - term3_min - term4_min - term5_min).real

    # @nb.jit(nopython=True, cache=True)
    def ker_C_n_pp_dd_s_t_integral_previous(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_phi, vert):
        s2 = s*s
        m_phi2 = m_phi*m_phi
        m_phi4 = m_phi2*m_phi2
        m_phi6 = m_phi4*m_phi2
        m_d2 = m_d*m_d
        m_d4 = m_d2*m_d2

        t_add = m_d2 + m_phi2
        t_min = t_add - 2.*E1*(E3 - p1*p3*ct_min/E1)
        t_max = t_add - 2.*E1*(E3 - p1*p3*ct_max/E1)
        t_m = t_add - 2.*E1*(E3 - p1*p3*ct_m/E1)
        t_p = t_add - 2.*E1*(E3 - p1*p3*ct_p/E1)
        t_mp = 2.*t_add - 2.*E1*(2.*E3 - p1*p3*(ct_m+ct_p)/E1)
        u_add = 2.*(m_d2+m_phi2) - s
        u_min = u_add - t_min
        u_max = u_add - t_max
        u_m = u_add - t_m
        u_p = u_add - t_p
        u_mp = u_m + u_p

        in_min_neq = (ct_min != ct_p)
        in_max_neq = (ct_max != ct_m)
        in_any_neq = np.logical_or(in_min_neq, in_max_neq)
        n = s.size

        min_1 = np.zeros(n)
        max_1 = np.zeros(n)

        prefac_2 = -(2.*m_d4*(6.*m_phi4-4.*m_phi2*s+s2)-8.*m_phi2*s*t_m*t_p+2.*s2*t_m*t_p-2.*m_phi6*t_mp
        +m_phi4*(12.*t_m*t_p+s*t_mp)+2.*m_d2*(2.*m_phi6+4.*m_phi2*s*t_mp-s2*t_mp-m_phi4*(s+6.*t_mp))) / \
        (2.*(s-2.*m_phi2)*np.sqrt(a * ((t_m-m_d2)**3.) * ((-t_p+m_d2)**3.)))
        min_2 = 0.5 * prefac_2 * pi
        max_2 = np.zeros(n)

        prefac_3 = -(2.*m_d4*(6.*m_phi4-4.*m_phi2*s+s2)-8.*m_phi2*s*u_m*u_p+2.*s2*u_m*u_p-2.*m_phi6*u_mp
        +m_phi4*(12.*u_m*u_p+s*u_mp)+2.*m_d2*(2.*m_phi6+4.*m_phi2*s*u_mp-s2*u_mp-m_phi4*(s+6.*u_mp))) / \
        (2.*(s-2.*m_phi2)*np.sqrt(a * ((-u_m+m_d2)**3.) * ((u_p-m_d2)**3.)))
        min_3 = 0.5 * prefac_3 * pi
        max_3 = np.zeros(n)

        log_part = -2.*pi/np.sqrt(-a)

        if np.any(in_min_neq):
            in_t_min_neq = np.logical_and(in_min_neq, (t_min != t_p))
            in_u_min_neq = np.logical_and(in_min_neq, (u_min != u_p))
            min_1[in_t_min_neq] = (m_phi4/(2.*a[in_t_min_neq]))*np.sqrt(a[in_t_min_neq]*(t_min[in_t_min_neq]-t_m[in_t_min_neq])*(t_min[in_t_min_neq]-t_p[in_t_min_neq]))*(1./((t_min[in_t_min_neq]-m_d2)*(t_m[in_t_min_neq]-m_d2)*(t_p[in_t_min_neq]-m_d2)) - 1./((u_min[in_t_min_neq]-m_d2)*(u_m[in_t_min_neq]-m_d2)*(u_p[in_t_min_neq]-m_d2)))
            min_2[in_t_min_neq] = prefac_2[in_t_min_neq] * np.arctan(np.sqrt((t_min[in_t_min_neq]-t_m[in_t_min_neq])*(m_d2-t_p[in_t_min_neq])/((t_min[in_t_min_neq]-t_p[in_t_min_neq])*(t_m[in_t_min_neq]-m_d2))))
            min_3[in_u_min_neq] = prefac_3[in_u_min_neq] * np.arctan(np.sqrt((u_min[in_u_min_neq]-u_m[in_u_min_neq])*(m_d2-u_p[in_u_min_neq])/((u_min[in_u_min_neq]-u_p[in_u_min_neq])*(u_m[in_u_min_neq]-m_d2))))
        if np.any(in_max_neq):
            in_t_max_neq = np.logical_and(in_max_neq, (t_max != t_m))
            in_u_max_neq = np.logical_and(in_max_neq, (u_max != u_m))
            max_1[in_t_max_neq] = (m_phi4/(2.*a[in_t_max_neq]))*np.sqrt(a[in_t_max_neq]*(t_max[in_t_max_neq]-t_m[in_t_max_neq])*(t_max[in_t_max_neq]-t_p[in_t_max_neq]))*(1./((t_max[in_t_max_neq]-m_d2)*(t_m[in_t_max_neq]-m_d2)*(t_p[in_t_max_neq]-m_d2)) - 1./((u_max[in_t_max_neq]-m_d2)*(u_m[in_t_max_neq]-m_d2)*(u_p[in_t_max_neq]-m_d2)))
            max_2[in_t_max_neq] = prefac_2[in_t_max_neq] * np.arctan(np.sqrt((t_max[in_t_max_neq]-t_m[in_t_max_neq])*(m_d2-t_p[in_t_max_neq])/((t_max[in_t_max_neq]-t_p[in_t_max_neq])*(t_m[in_t_max_neq]-m_d2))))
            max_3[in_u_max_neq] = prefac_3[in_u_max_neq] * np.arctan(np.sqrt((u_max[in_u_max_neq]-u_m[in_u_max_neq])*(m_d2-u_p[in_u_max_neq])/((u_max[in_u_max_neq]-u_p[in_u_max_neq])*(u_m[in_u_max_neq]-m_d2))))
        if np.any(in_any_neq):
            c_zero = complex(0.,0.)
            log_part[in_any_neq] = -(-4.*np.log((np.sqrt(ct_max[in_any_neq]-ct_m[in_any_neq]+c_zero)+np.sqrt(ct_max[in_any_neq]-ct_p[in_any_neq]+c_zero))/(np.sqrt(ct_min[in_any_neq]-ct_m[in_any_neq]+c_zero)+np.sqrt(ct_min[in_any_neq]-ct_p[in_any_neq]+c_zero)))/np.sqrt(a[in_any_neq]+c_zero)).real

        # def kernel(ct):
        #     Epdiff = E1*(E3-p1*p3*ct/E1)
        #     num = -((s-4.*Epdiff)**2.)*(4.*Epdiff*Epdiff+s*(m_phi2-2.*Epdiff))
        #     den = 2.*sqrt(a*(ct-ct_m)*(ct-ct_p))*((m_phi2-2.*Epdiff)**2.)*((m_phi2+2.*Epdiff-s)**2.)
        #     return num/den if den != 0. else 0.
        # res, err = quad(kernel, ct_min, ct_max, points=(ct_m, ct_p), epsabs=0., epsrel=1e-3)
        # print(res/ct_int)

        return 4.*vert*(max_1 + max_2 + max_3 - min_1 - min_2 - min_3 + log_part)

    # Just for comparison to XX --> dd
    @nb.jit(nopython=True, cache=True)
    def sigma_pp_dd(s, m_d, m_phi, vert):
        m_d2 = m_d*m_d
        m_phi2 = m_phi*m_phi
        m_phi4 = m_phi2*m_phi2
        if s <= 4.*m_phi2 or s <= 4.*m_d2:
            return 0.
        
        p3cm = sqrt(0.25*s - m_phi2)
        p1cm = sqrt(0.25*s - m_d2)
        t_upper = -(p1cm - p3cm)**2 + 0j
        t_lower = -(p1cm + p3cm)**2 + 0j

        # Anton: Result from Depta, with switched propagator momentum 
        M2_int_upper_Depta = -2*(-(((6*m_phi4-4*m_phi2*s+s**2)*(np.log(m_d2-t_upper)-np.log(2*m_phi2+m_d2-s-t_upper)))/(2*m_phi2-s))+m_phi4*(1/(2*m_phi2+m_d2-s-t_upper)+1/(m_d2-t_upper))-4*m_d2+4*t_upper)
        
        M2_int_lower_Depta = -2*(-(((6*m_phi4-4*m_phi2*s+s**2)*(np.log(m_d2-t_lower)-np.log(2*m_phi2+m_d2-s-t_lower)))/(2*m_phi2-s))+m_phi4*(1/(2*m_phi2+m_d2-s-t_lower)+1/(m_d2-t_lower))-4*m_d2+4*t_lower)

        M2_int_Depta = M2_int_upper_Depta - M2_int_lower_Depta
        sigma_Depta = vert*M2_int_Depta.real/(2*64.*pi*s*p1cm*p1cm) 

        # Anton: Same calculation as above, with propagator momentum the correct order (relative minus sign)
        M2_int_upper = 2*((m_phi2-4*m_d2)**2/(-2*m_phi2-m_d2+s+t_upper)-(m_phi2-4*m_d2)**2/(m_d2-t_upper)+((6*m_phi4-4*m_phi2*(4*m_d2+s)-32*m_d2**2+16*m_d2*s+s**2)*(np.log(t_upper-m_d2)-np.log(-2*m_phi2-m_d2+s+t_upper)))/(2*m_phi2-s)-4*t_upper)

        M2_int_lower = 2*((m_phi2-4*m_d2)**2/(-2*m_phi2-m_d2+s+t_lower)-(m_phi2-4*m_d2)**2/(m_d2-t_lower)+((6*m_phi4-4*m_phi2*(4*m_d2+s)-32*m_d2**2+16*m_d2*s+s**2)*(np.log(t_lower-m_d2)-np.log(-2*m_phi2-m_d2+s+t_lower)))/(2*m_phi2-s)-4*t_lower)

        M2_int = M2_int_upper - M2_int_lower
        sigma = vert*M2_int.real/(64.*pi*s*p1cm*p1cm) 

        # Anton: Divide by 2 for identical particles in final state
        return sigma / 2
        return sigma_Depta / 2

    ker_C_n_XX_dd_s_t_integral_val = ker_C_n_XX_dd_s_t_integral(ct_min=ct_min[in_res], ct_max=ct_max[in_res], ct_p=ct_p[in_res], ct_m=ct_m[in_res], a=a[in_res], s=s[in_res], E1=E1, E3=E3, p1=p1, p3=p3, m_d=m_d, m_X=m_X, vert=1)

    # ker_C_n_pp_dd_s_t_integral_val = ker_C_n_pp_dd_s_t_integral(ct_min=ct_min[in_res], ct_max=ct_max[in_res], ct_p=ct_p[in_res], ct_m=ct_m[in_res], a=a[in_res], s=s[in_res], E1=E1, E3=E3, p1=p1, p3=p3, m_d=m_d, m_phi=m_X, vert=1)

    # ker_C_n_pp_dd_s_t_integral_val_Depta = ker_C_n_pp_dd_s_t_integral_Depta(ct_min=ct_min[in_res], ct_max=ct_max[in_res], ct_p=ct_p[in_res], ct_m=ct_m[in_res], a=a[in_res], s=s[in_res], E1=E1, E3=E3, p1=p1, p3=p3, m_d=m_d, m_phi=m_X, vert=1)

    ker_C_n_pp_dd_s_t_integral_val_previous = ker_C_n_pp_dd_s_t_integral_previous(ct_min=ct_min[in_res], ct_max=ct_max[in_res], ct_p=ct_p[in_res], ct_m=ct_m[in_res], a=a[in_res], s=s[in_res], E1=E1, E3=E3, p1=p1, p3=p3, m_d=m_d, m_phi=m_X, vert=1)
    print(f'ker_C_n_XX_dd_s_t_integral ran in {time.time()-time1}s')
    # print(ker_C_n_XX_dd_s_t_integral_val)
    # C_n_XX_dd_val = C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, vert, type=0)
    # print(C_n_XX_dd_val)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # x = ln(s/s_min) / ln(s_max/s_min)
    ax.plot(s[in_res], ker_C_n_XX_dd_s_t_integral_val, 'r', label='XX_dd')
    # ax2.plot(s[in_res], ker_C_n_pp_dd_s_t_integral_val, 'tab:blue', label='pp_dd')
    # ax2.plot(s[in_res], ker_C_n_XX_dd_s_t_integral_val + ker_C_n_XX_dd_s_t_integral_val_2)
    # ax2.plot(s[in_res], ker_C_n_pp_dd_s_t_integral_val_Depta, 'tab:green', label='pp_dd_Depta')
    ax2.plot(s[in_res], ker_C_n_pp_dd_s_t_integral_val_previous, 'tab:orange', label='pp_dd_previous')
    ax.set_xscale('log')
    ax2.set_xscale('log')
            
    ax.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()

    s_min = 4*m_d**2
    S = 10**(np.linspace(np.log10(s_min), 4, int(1e3)))
    # S = np.linspace(s_min, 1e4, int(1e3))
    m1 = m_d
    m2 = m_d
    m3 = m_X
    m4 = m_X
    E1cm = (S + m1*m1 - m2*m2) / (2*np.sqrt(S))
    E3cm = (S + m3*m3 - m4*m4) / (2*np.sqrt(S))
    p1cm = np.sqrt((E1cm - m1)*(E1cm + m1))
    p3cm = np.sqrt((E3cm - m3)*(E3cm + m3))

    E13diff = (m1*m1 - m2*m2 - m3*m3 + m4*m4) / (2*np.sqrt(S))
    # Anton: This is the restricted kinematical region for t.
    # Outside this region, |M|^2 may become negative 
    T_min = (E13diff + (p1cm + p3cm))*(E13diff - (p1cm + p3cm))
    T_max = (E13diff + (p1cm - p3cm))*(E13diff - (p1cm - p3cm))
    # T = np.linspace(np.nanmin(T_min), np.nanmax(T_max), int(1e3))
    T = np.linspace(-1, 1, int(1e3))
    # T = -10**(np.linspace(2, 0, int(1e3)))
    s, t = np.meshgrid(S, T, indexing='ij')
    # plt.plot(S, np.linspace(T_min, T_max, int(1e3)))
    # plt.show()

    def M2_bhaba_massless(s, t, vert):
        u = -s - t
        M2 = (u**2 + s**2)/t**2 + 2*u**2/(s*t) + (u**2 + t**2)/s**2
        return vert*M2
    
    def M2_electron_positron_to_2photon(s, t, m_d2, vert):
        """
        https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-GaGa
        or PS. Eq. 5.105
        Expression for dd --> XX reduce to this in mX --> 0 limit, so should be correct. 
        Worried since M2 becomes negative for certain s (u), t values..
        """
        u = 2*m_d2 - s - t
        t2 = t*t
        t3 = t*t2
        u2 = u*u
        u3 = u*u2

        m_d4 = m_d2*m_d2
        m_d8 = m_d4*m_d4
        M2 = -((8*(6*m_d8-m_d4*(3*t2+14*t*u+3*u2)+m_d2*(t3+7*t2*u+7*t*u2+u3)-t*u*(t2+u2)))/((m_d2-t)**2*(m_d2-u)**2))
        return vert*M2
    
    def M2_pp_dd(s, t, m_d2, vert, m_phi2):
        m_d4 = m_d2*m_d2
        m_d6 = m_d2*m_d4
        m_d8 = m_d4*m_d4
        m_phi4 = m_phi2*m_phi2
        m_phi6 = m_phi2*m_phi4
        s2 = s*s
        s3 = s*s2
        t2 = t*t
        t3 = t*t2
        # Anton: M2 found by Mathematica
        M2 = -(1/((m_d2-t)**2*(-m_d2-2*m_phi2+s+t)**2))*2*(4*m_d8+8*m_d6*(-2*m_phi2+s-2*t)+m_d4*(24*m_phi4+4*m_phi2*(4*t-5*s)+5*s2-8*s*t+24*t2)-m_d2*(16*m_phi6-16*m_phi4*(s+t)+2*m_phi2*(s2+12*s*t-8*t2)+s3-6*s2*t+8*s*t2+16*t3)+(-2*m_phi2+s+2*t)**2*(m_phi4-2*m_phi2*t+t*(s+t)))
        # Anton: Analytical calculation, is equal to M2, M2me = M2 
        M2me = (4*(4*m_d2*(s-2*m_phi2)*((t-m_d2)*(m_phi2+3*m_d2-s-t)+(-m_phi2+m_d2+t)*(2*m_phi2+m_d2-s-t))+1/2*(2*m_phi2+2*m_d2-s-2*t)**2*((t-m_d2)*(2*m_phi2+m_d2-s-t)-m_phi4)))/((t-m_d2)**2*(2*m_phi2+m_d2-s-t)**2)
        # Anton: Found by Depta -- propagator momentum wrong 
        M2Depta = (4*(1/2*(2*m_phi2+2*m_d2-s-2*t)**2*((t-m_d2)*(2*m_phi2+m_d2-s-t)-m_phi4)))/((t-m_d2)**2*(2*m_phi2+m_d2-s-t)**2)
        return vert*M2
        return vert*M2Depta
        return vert*(M2me - M2Depta)

    S_fixed = 1.5e2
    plt.plot(T, M2_XX_dd(S_fixed, T, m_d2=m_d**2, vert=vert_el, m_X2=m_X**2), 'k')
    plt.plot(T, M2_electron_positron_to_2photon(S_fixed, T, m_d2=m_d**2, vert=vert_el), 'r--')
    plt.plot(T, M2_pp_dd(S_fixed, T, m_d2=m_d**2, vert=vert_el, m_phi2=m_X**2), 'tab:green')
    plt.show()

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    
    M2_XX_dd_val = M2_XX_dd(s, t, m_d2=m_d**2, vert=vert_el, m_X2=m_X**2)
    M2_pp_dd_val = M2_pp_dd(s, t, m_d2=m_d**2, vert=vert_el, m_phi2=m_X**2)
    M2_bhaba_massless_val = M2_bhaba_massless(s, t, vert_el)
    M2_electron_positron_to_2photon_val = M2_electron_positron_to_2photon(s, t, m_d2=m_d**2, vert=vert_el)
    plot_XX_dd = ax1.contourf(s, t, np.log10(M2_XX_dd_val), levels=300, cmap='jet')
    plot_compare = ax2.contourf(s, t, np.log10(M2_pp_dd_val), levels=300, cmap='jet')
    # plot_compare = ax2.contourf(s, t, np.log10(M2_bhaba_massless_val), levels=300, cmap='jet')
    # plot_compare = ax2.contourf(s, t, np.log10(M2_electron_positron_to_2photon_val), levels=300, cmap='jet')
    fig1.colorbar(plot_XX_dd)
    fig2.colorbar(plot_compare)
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()

    from matplotlib import collections  as mc

    fig, ax = plt.subplots()

    vert_XX_dd = y**4 * np.cos(th)**8
    s_sigma = 10**(np.linspace(np.log10(s_min), 2, int(1e4)))
    sigma_XX_dd_val = np.vectorize(sigma_XX_dd)(s=s_sigma, m_d=m_d, m_X=m_X, vert=vert_XX_dd)
    sigma_pp_dd_val = np.vectorize(sigma_pp_dd)(s=s_sigma, m_d=m_d, m_phi=m_X, vert=vert_XX_dd)

    # Color lines based on which index element has, for debugging purposes
    # lines = [np.column_stack([[s[i], s[i+1]], [sigma[i], sigma[i+1]]]) for i in range(len(s)-1)]
    # lc = mc.LineCollection(lines[:int(x.size*0.98)], cmap="jet", lw=2)       # Avoid double-plotting from s by slicing ~ 990
    # # print(np.where(np.diff(s)<0))
    # # Set the values used to determine the color
    # lc.set_array(range(len(s)))
    # ax_collection = ax.add_collection(lc)
    # colorbar = fig.colorbar(ax_collection, ax=ax)

    # ax.plot(s, sigma*s*(s - 4*m_X**2), linestyle='none')        # |M|^2 integrated over t 
    ax.axvline(4*m_X**2, color='k', linestyle='--')
    ax.plot(s_sigma, sigma_XX_dd_val, color='r', label='sigma vector')
    ax.plot(s_sigma, sigma_pp_dd_val, color='tab:blue', label='sigma scalar')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

    res_XX_dd = C_n_XX_dd(m_d=m_d, m_X=m_X, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, type=0)
    res_pp_dd = C_n_pp_dd(m_d=m_d, m_phi=m_X, k_d=k_d, k_phi=k_X, T_d=T_d, xi_d=xi_d, xi_phi=xi_X, vert=vert_el, type=0)
    res_34_12 = C_34_12(type=0, nFW=1, nBW=-1, m1=m_d, m2=m_a, m3=m_d, m4=m_d, k1=k_d, k2=k_a, k3=k_d, k4=k_d, T1=T_d, T2=T_a, T3=T_d, T4=T_d, xi1=xi_d, xi2=xi_a, xi3=xi_d, xi4=xi_d, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2, res_sub=False, thermal_width=True)
    print(res_XX_dd)
    print(res_pp_dd)
    print(res_34_12)
