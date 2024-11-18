#! /usr/bin/env python3

import numpy as np
import numba as nb
from math import sin, cos, sqrt, atan, log
from scipy.integrate import quad

rtol_int = 1e-4

@nb.jit(nopython=True, cache=True)
def Gamma_phi(y, th, m_X, m_d):
    y2 = y*y
    sth = sin(th)
    cth = cos(th)

    # Decay to aa, ad, and dd
    X_aa = y2*sth**4 * m_X/(8*np.pi)
    X_ad = y2*cth**2*sth**2 * (2*m_X**4 - m_X**2*m_d**2 - m_d**4)*(m_X**2 - m_d**2)/(8*np.pi*m_X**5)*(m_X**2 > m_d**2)
    X_dd = y2*cth**4 * np.sqrt(m_X**2 - 4*m_d**2)*(m_X**2 + 2*m_d**2)/(8*np.pi*m_X**2)*(m_x**2 > 2*m_d**2)

    return X_aa + X_ad + X_dd

# sub indicates if s-channel on-shell resonance is subtracted
@nb.jit(nopython=True, cache=True)
def M2_gen(s, t, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=False):
    """
    12 --> 34, 1,2,3,4 = a, d
    """
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    m42 = m4*m4
    m13 = m12*m1
    m23 = m22*m2
    m33 = m32*m3
    m43 = m42*m4

    u = m12 + m22 + m32 + m42 - s - t

    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    t_prop = 1. / ((t-m_phi2)*(t-m_phi2) + m_Gamma_phi2)
    u_prop = 1. / ((u-m_phi2)*(u-m_phi2) + m_Gamma_phi2)

    ss = ((m1+m2)*(m1+m2)-s)*((m3+m4)*(m3+m4)-s)*s_prop*(s_prop*(s-m_phi2)*(s-m_phi2) if sub else 1.)
    tt = ((m1+m3)*(m1+m3)-t)*((m2+m4)*(m2+m4)-t)*t_prop
    uu = ((m1+m4)*(m1+m4)-u)*((m2+m3)*(m2+m3)-u)*u_prop
    st = -(m23*m3+m13*m4+m22*m3*(m3+m4)+m12*m4*(m2+m3+m4)-s*(m3*m4+t)
     +m1*(m22*m3+m3*m42+m43+m2*(m32+2.*m3*m4+m42-s)-m4*s-m3*t-m4*t)
     +m2*(m33+m32*m4-m4*t-m3*(s+t)))*s_prop*t_prop*((s-m_phi2)*(t-m_phi2)+(0. if sub else m_Gamma_phi2))
    su = -(m13*m3+m23*m4+m22*m4*(m3+m4)+m12*m3*(m2+m3+m4)-s*(m3*m4+u)
     +m1*(m33+m22*m4+m32*m4+m2*(m32+2.*m3*m4+m42-s)-m4*u-m3*(s+u))
     +m2*(m3*m42+m43-m4*s-m3*u-m4*u))*s_prop*u_prop*((s-m_phi2)*(u-m_phi2)+(0. if sub else m_Gamma_phi2))
    tu = -((m13*m2+m33*m4+m42*m3*(m3+m4)+m12*m2*(m2+m3+m4)-m3*m4*(t+u)-t*u
     +m1*(m23+m32*m4+m3*m42+m22*(m3+m4)-m3*t+m2*(2.*m3*m4-t-u)-m4*u)
     +m2*(m32*m4+m3*m42-m4*t-m3*u))
     *((t-m_phi2)*(u-m_phi2)+m_Gamma_phi2))*t_prop*u_prop

    return 4.*vert*ss#(ss+tt+uu+st+su+tu)

@nb.jit(nopython=True, cache=True)
def M2_gen_ss(s, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2):
    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    ss = (((m1+m2)**2.)-s)*(((m3+m4)**2.)-s)*s_prop
    return 4.*vert*ss

@nb.jit(nopython=True, cache=True)
def M2_fi(s, t, m_d2, vert, m_phi2, m_Gamma_phi2):
    """
    aa --> dd
    """
    u = 2.*m_d2 - s - t

    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    t_prop = 1. / ((t-m_phi2)*(t-m_phi2) + m_Gamma_phi2)
    u_prop = 1. / ((u-m_phi2)*(u-m_phi2) + m_Gamma_phi2)

    ss = s * (s-4.*m_d2) * s_prop
    tt = (t - m_d2) * (t - m_d2) * t_prop
    uu = (u - m_d2) * (u - m_d2) * u_prop
    st = s * (t + m_d2) * ((s-m_phi2)*(t-m_phi2) + m_Gamma_phi2) * s_prop * t_prop
    su = s * (u + m_d2) * ((s-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * s_prop * u_prop
    tu = - (3.*m_d2*m_d2 - t*u - m_d2*(t+u)) * ((t-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * t_prop * u_prop

    return 4.*vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_tr(s, t, m_d2, vert, m_phi2, m_Gamma_phi2):
    """
    as --> dd
    """
    u = 3.*m_d2 - s - t
    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    t_prop = 1. / ((t-m_phi2)*(t-m_phi2) + m_Gamma_phi2)
    u_prop = 1. / ((u-m_phi2)*(u-m_phi2) + m_Gamma_phi2)
    ss = (4.*m_d2*m_d2 - 5.*m_d2*s + s*s) * s_prop
    tt = (4.*m_d2*m_d2 - 5.*m_d2*t + t*t) * t_prop
    uu = (4.*m_d2*m_d2 - 5.*m_d2*u + u*u) * u_prop
    st = - (5.*m_d2*m_d2 - s*t - 2.*m_d2*(s+t)) * ((s-m_phi2)*(t-m_phi2) + m_Gamma_phi2) * s_prop * t_prop
    su = - (5.*m_d2*m_d2 - s*u - 2.*m_d2*(s+u)) * ((s-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * s_prop * u_prop
    tu = - (5.*m_d2*m_d2 - t*u - 2.*m_d2*(t+u)) * ((t-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * t_prop * u_prop

    return 4.*vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_el(s, t, m_d, vert, m_phi2, m_Gamma_phi2):
    """
    dd --> dd
    """
    return M2_gen(s, t, m_d, vert, m_phi2, m_Gamma_phi2)

# Cross-sections for each process el, fi, gen

@nb.jit(nopython=True, cache=True)
def ker_sigma_gen(t, s, p1cm, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub):
    return M2_gen(s, t, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=sub)/(64.*np.pi*s*p1cm*p1cm)

# no factor taking care of identical particles (not known on this level)
def sigma_gen(s, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=False):
    if s < (m1+m2)**2. or s < (m3+m4)**2.:
        return 0.
    sqrt_s = sqrt(s)
    E1cm = (s+m1*m1-m2*m2)/(2.*sqrt_s)
    p1cm = sqrt((E1cm-m1)*(E1cm+m1))
    E3cm = (s+m3*m3-m4*m4)/(2.*sqrt_s)
    p3cm = sqrt((E3cm-m3)*(E3cm+m3))

    a = (m1*m1-m3*m3-m2*m2+m4*m4)/(2.*sqrt_s)
    t0 = (a-(p1cm-p3cm))*(a+(p1cm-p3cm))
    t1 = (a-(p1cm+p3cm))*(a+(p1cm+p3cm))
    # t0 = ((m1*m1-m3*m3-m2*m2+m4*m4)**2.)/(4.*s) - ((p1cm-p3cm)**2.)
    # t1 = ((m1*m1-m3*m3-m2*m2+m4*m4)**2.)/(4.*s) - ((p1cm+p3cm)**2.)

    res, err = quad(ker_sigma_gen, t1, t0, args=(s, p1cm, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub), epsabs=0., epsrel=rtol_int)

    return res

@nb.jit(nopython=True, cache=True)
def sigma_tr(s, m_d2, vert, m_phi2, m_Gamma_phi2):
    if s < 4.*m_d2:
        return 0.
    if s > 1e6*m_d2 and s > 1e6*m_phi2:
        return 0.5*1.5*vert/(np.pi*s) # factor 0.5 due to identical particles in final state
    m_phi4 = m_phi2 * m_phi2
    m_d4 = m_d2 * m_d2
    s2 = s*s
    m_Gamma_phi = sqrt(m_Gamma_phi2)
    sqrt_fac = sqrt(s*(s-4.*m_d2))

    fac_atan = 2.*(-3.*m_d2+2.*m_phi2+s)*((4.*m_d4-5.*m_d2*m_phi2+m_phi4)*(s-m_phi2)*(s-m_phi2)
     -m_Gamma_phi2*(m_d4+m_d2*m_phi2+m_phi4-6.*m_phi2*s+3.*s2)-2.*m_Gamma_phi2*m_Gamma_phi2)/m_Gamma_phi
    sum_atan = fac_atan*atan((m_Gamma_phi*sqrt_fac*(s-m_d2))/(m_d4*m_d2-3.*m_d2*m_phi2*s+m_Gamma_phi2*s+m_phi4*s+m_phi2*s2))

    fac_log = (-5.*m_phi4*m_phi4 + 15.*m_d4*m_d2*(m_phi2 - s) - 4.*m_phi2*s2*s
     + m_phi4*(9.*s*m_phi2 - 4.*m_Gamma_phi2) - 7.*s*m_phi2*m_Gamma_phi2 + m_Gamma_phi2*m_Gamma_phi2
     - m_d4*(30.*m_phi4 + 3.*s2 - 33.*s*m_phi2 + 8.*m_Gamma_phi2)
     + m_d2*(23.*m_phi4*m_phi2 + s2*s + 12.*s2*m_phi2 + 4.*s*m_Gamma_phi2
     - 36.*s*m_phi4 + 15.*m_phi2*m_Gamma_phi2))
    sum_log = fac_log * log(
     (-2.*m_d4*m_d2+2.*m_phi4*s+s2*s+s2*sqrt_fac+3.*m_d4*(3.*s+sqrt_fac)
      -2.*m_d2*(3.*s2+2.*s*sqrt_fac+m_phi2*(3.*s+sqrt_fac))+2.*(m_phi2*s*sqrt_fac+m_phi2*s2+m_Gamma_phi2)) /
     (-2.*m_d4*m_d2+2.*m_phi4*s+s2*s-s2*sqrt_fac+3.*m_d4*(3.*s-sqrt_fac)
      +2.*m_d2*(-3.*s2+2.*s*sqrt_fac+m_phi2*(-3.*s+sqrt_fac))+2.*(-m_phi2*s*sqrt_fac+m_phi2*s2+m_Gamma_phi2)))

    sum_3 = (m_d2-s)*(3.*m_d2-2.*m_phi2-s)*(4.*m_d4+3.*m_phi4+6.*s2-m_d2*(s+4.*m_phi2)-8.*s*m_phi2+3.*m_Gamma_phi2)*sqrt_fac/s

    # factor 0.5 due to identical particles in final state
    return 0.5*vert*(sum_atan+sum_log+sum_3)/(4.*np.pi*(s-m_d2)*(s-m_d2)*(s-3.*m_d2+2.*m_phi2)*((s-m_phi2)*(s-m_phi2)+m_Gamma_phi2))

@nb.jit(nopython=True, cache=True)
def sigma_el(s, m_d2, vert, m_phi2, m_Gamma_phi2):
    if s < 4.*m_d2:
        return 0.
    if s > 1e6*m_d2 and s > 1e6*m_phi2:
        return 0.5*1.5*vert/(np.pi*s) # factor 0.5 due to identical particles in final state
    m_phi4 = m_phi2 * m_phi2
    m_d4 = m_d2 * m_d2
    s2 = s*s
    m_Gamma_phi = sqrt(m_Gamma_phi2)

    fac_atan = 2.*(-4.*m_d2+2.*m_phi2+s)*(-(((m_phi2-4.*m_d2)*(s-m_phi2))**2.)+m_Gamma_phi2*(m_phi4-6.*m_phi2*s+3.*s2)+2.*m_Gamma_phi2*m_Gamma_phi2)
    sum_atan = fac_atan*atan(m_Gamma_phi*(4.*m_d2-s)/(-4.*m_d2*m_phi2+m_phi4+s*m_phi2+m_Gamma_phi2))

    fac_log = m_Gamma_phi*(5.*m_phi4*m_phi4+4.*m_phi2*s2*s+64.*m_d4*m_d2*(s-m_phi2)+16.*m_d4*(5.*m_phi4-5.*s*m_phi2+m_Gamma_phi2)
     +m_phi4*(-9.*m_phi2*s+4.*m_Gamma_phi2)+7.*s*m_phi2*m_Gamma_phi2-m_Gamma_phi2*m_Gamma_phi2
     -4.*m_d2*(9.*m_phi4*m_phi2+s*(4.*s*m_phi2+m_Gamma_phi2)+m_phi2*(-13.*s*m_phi2+5.*m_Gamma_phi2)))
    sum_log = fac_log*log((m_phi4+m_Gamma_phi2)/(m_Gamma_phi2+((s-4.*m_d2+m_phi2)**2.)))

    sum_3 = m_Gamma_phi*(4.*m_d2-s)*(4.*m_d2-2.*m_phi2-s)*(16.*m_d4-8.*m_d2*m_phi2+3.*m_phi4+6.*s2-8.*s*m_phi2+3.*m_Gamma_phi2)

    # factor 0.5 due to identical particles in final state
    return 0.5*vert*(sum_atan+sum_log+sum_3)/(4.*np.pi*m_Gamma_phi*s*(s-4.*m_d2)*(s-4.*m_d2+2.*m_phi2)*(m_phi4+s2-2.*s*m_phi2+m_Gamma_phi2))
