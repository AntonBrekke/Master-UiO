#! /usr/bin/env python3

import numpy as np
import numba as nb
from math import sin, cos, sqrt, atan, log
from scipy.integrate import quad, quad_vec

rtol_int = 1e-4

"""
Changes:
Updated 
* Gamma_phi to Gamma_X 
* M2_tr
* M2_gen
* M2_el
* M2_fi
* sigma_gen

What must be done: 
Update 
M2_gen_ss, sigma_fi, sigma_tr, sigma_el

However, only sigma_gen is ever called in the code - 
may not have to fix the others 
"""

@nb.jit(nopython=True, cache=True)
def Gamma_X(y, th, m_X, m_d):
    y2 = y*y
    sth = np.sin(th)
    cth = np.cos(th)

    # Decay to aa, ad, and dd. Have used m_a = 0.
    X_aa = y2*sth**4 * m_X/(8*np.pi)
    X_ad = y2*cth**2*sth**2/(8*np.pi*m_X**5)*(2*m_X**4 - m_X**2*m_d**2 - m_d**4)*(m_X**2 - m_d**2) * (m_X**2 > m_d**2)
    X_dd = y2*cth**4/(8*np.pi*m_X**2)*np.sqrt(m_X**2 - 4*m_d**2)*(m_X**2 + 2*m_d**2) * (m_X**2 > 2*m_d**2)

    return X_aa + X_ad + X_dd

# sub indicates if s-channel on-shell resonance is subtracted
@nb.jit(nopython=True, cache=True)
def M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False):
    """
    12 --> 34, 1,2,3,4 = a, d
    Rather large
    """
    m12 = m1*m1
    m13 = m1*m12
    m14 = m12*m12
    m15 = m12*m13
    m16 = m12*m14
    # m17 = m13*m14
    m18 = m14*m14

    m22 = m2*m2
    m23 = m2*m22
    m24 = m22*m22
    m25 = m22*m23
    m26 = m22*m24
    # m27 = m23*m24
    m28 = m24*m24

    m32 = m3*m3
    m33 = m3*m32
    m34 = m32*m32
    m35 = m32*m33
    # m36 = m32*m34
    # m37 = m33*m34
    # m38 = m34*m34

    m42 = m4*m4
    m43 = m4*m42
    m44 = m42*m42
    m45 = m42*m43
    # m46 = m42*m44
    # m47 = m43*m44
    # m48 = m44*m44

    m_X4 = m_X2*m_X2
    # m_X6 = m_X2*m_X4
    # m_X8 = m_X4*m_X4

    u = m12 + m22 + m32 + m42 - s - t

    s2 = s*s
    s3 = s*s2
    t2 = t*t
    t3 = t*t2
    u2 = u*u
    u3 = u*u2

    s_prop2 = 1. / ((s - m_X2)*(s - m_X2) + m_Gamma_X2)
    t_prop2 = 1. / ((t - m_X2)*(t - m_X2) + m_Gamma_X2)
    u_prop2 = 1. / ((u - m_X2)*(u - m_X2) + m_Gamma_X2)
    
    ss = s_prop2*s_prop2*(1/(m_X4))*4*(m_Gamma_X2+(m_X2-s)**2)*(-m18+(-2*m32-2*m42+s+2*t+2*u)*m16+(4*m_X2*m2-2*m2*s)*m15+(2*m24+(2*m32+2*m42+3*s-2*t-2*u)*m22-4*m32*m42+s2-t2-u2+m32*s+m42*s+2*m3*m4*s+2*m32*t+2*m42*t-2*s*t-2*m_X2*(4*m22+2*m3*m4+s-t-u)+2*m32*u+2*m42*u-2*s*u-2*t*u)*m14+4*m2*(2*m_X2-s)*(m22+m32+m42-t-u)*m13+(2*(2*m22+m32+m42-2*m3*m4-t-u)*m_X4-2*(4*m24+(8*m32-4*m4*m3+8*m42-2*(s+3*(t+u)))*m22-s2+t2+u2+m42*s-2*m3*m4*s-m42*t+m32*(4*m42+s-3*t-u)-3*m42*u+2*t*u)*m_X2+m24*(2*m32+2*m42+3*s-2*t-2*u)+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u)))+2*m22*((4*m42+3*s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2-2*s*t-2*s*u+2*t*u+m42*(3*s-2*(t+u))))*m12-2*m2*(2*(m32-4*m4*m3+m42-s)*m_X4-2*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u)))*m_X2+s*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u))))*m1-m28+m26*(-2*m32-2*m42+s+2*t+2*u)+2*m_X4*((2*m42-t-u)*m32+2*m4*s*m3+t2+u2-m42*(t+u))+m24*(-2*(2*m3*m4+s-t-u)*m_X2+s2-t2-u2+m42*s+2*m3*m4*s+2*m42*t-2*s*t+2*m42*u-2*s*u-2*t*u+m32*(-4*m42+s+2*(t+u)))+m22*(2*(m32-2*m4*m3+m42-t-u)*m_X4-2*((4*m42+s-t-3*u)*m32-2*m4*s*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m_X2+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u)))))

    tt = t_prop2*t_prop2*(1/(m_X4))*4*(m_Gamma_X2+(m_X2-t)**2)*(-m18+(-2*m22-2*m42+2*s+t+2*u)*m16+(4*m_X2*m3-2*m3*t)*m15+(2*m34+2*m42*m32-2*s*m32+3*t*m32-2*u*m32-s2+t2-u2+2*m42*s+m42*t+2*m2*m4*t-2*s*t-2*m_X2*(4*m32+2*m2*m4-s+t-u)+2*m42*u-2*s*u-2*t*u+m22*(2*m32-4*m42+2*s+t+2*u))*m14+4*m3*(2*m_X2-t)*(m22+m32+m42-s-u)*m13+(2*(m22-2*m4*m2+2*m32+m42-s-u)*m_X4-2*(4*m34+(8*m42-2*(3*s+t+3*u))*m32+s2-t2+u2-m42*s+m42*t-2*m2*m4*(2*m32+t)+m22*(8*m32+4*m42-3*s+t-u)-3*m42*u+2*s*u)*m_X2-t3+2*m34*m42+2*m32*s2-2*m32*t2+m42*t2+2*m32*u2+t*u2-2*m34*s-4*m32*m42*s+3*m34*t+6*m32*m42*t+s2*t-4*m32*s*t-2*m42*s*t-2*m2*m4*t*(2*m32+t)+m22*(2*m34+(8*m42-4*s+6*t-4*u)*m32+t*(4*m42-2*s+t-2*u))-2*m34*u-4*m32*m42*u+4*m32*s*u-4*m32*t*u-2*m42*t*u+2*s*t*u)*m12-2*m3*(2*(m22-4*m4*m2+m42-t)*m_X4-2*(m34+2*(m42-s-u)*m32+s2-t2+u2-2*m42*s+m42*t-2*m2*m4*t+m22*(2*m32+4*m42-2*s+t-2*u)-2*m42*u+2*s*u)*m_X2+t*(m34+2*(m42-s-u)*m32+s2-t2+u2-2*m42*s+m42*t-2*m2*m4*t+m22*(2*m32+4*m42-2*s+t-2*u)-2*m42*u+2*s*u))*m1+2*m_X4*((m32+2*m42-s-u)*m22+2*m4*(t-m32)*m2+s2+u2-m42*s+m32*(m42-s-u)-m42*u)-m32*(m32-t)*(m34+2*(m42-s-u)*m32+s2-t2+u2-2*m42*s+m42*t-2*m2*m4*t+m22*(2*m32+4*m42-2*s+t-2*u)-2*m42*u+2*s*u)-2*m_X2*m32*((4*m42-s+t-3*u)*m22+2*m4*(m32-t)*m2+s2-t2+u2-3*m42*s+m42*t-m42*u+2*s*u-m32*(s-t+u)))

    uu = u_prop2*u_prop2*(1/(m_X4))*4*(m_Gamma_X2+(m_X2-u)**2)*(-m18+(-2*m22-2*m32+2*s+2*t+u)*m16+(4*m_X2*m4-2*m4*u)*m15+(2*m44+2*m32*m42-2*s*m42-2*t*m42+3*u*m42-s2-t2+u2+2*m32*s+2*m32*t-2*s*t+m32*u+2*m2*m3*u-2*s*u-2*t*u-2*m_X2*(4*m42+2*m2*m3-s-t+u)+m22*(-4*m32+2*m42+2*s+2*t+u))*m14+4*m4*(m22+m32+m42-s-t)*(2*m_X2-u)*m13+(2*(m22-2*m3*m2+m32+2*m42-s-t)*m_X4-2*(4*m44-6*s*m42-6*t*m42-2*u*m42+s2+t2-u2+2*s*t-2*m2*m3*(2*m42+u)+m32*(8*m42-s-3*t+u)+m22*(4*m32+8*m42-3*s-t+u))*m_X2+2*m32*m44-u3+2*m42*s2+2*m42*t2+m32*u2-2*m42*u2-2*m44*s-4*m32*m42*s-2*m44*t-4*m32*m42*t+4*m42*s*t+3*m44*u+6*m32*m42*u+s2*u+t2*u-2*m32*s*u-4*m42*s*u-2*m32*t*u-4*m42*t*u+2*s*t*u-2*m2*m3*u*(2*m42+u)+m22*(2*m44+(-4*s-4*t+6*u)*m42+4*m32*(2*m42+u)+u*(-2*s-2*t+u)))*m12-2*m4*(2*(m22-4*m3*m2+m32-u)*m_X4-2*(m44-2*s*m42-2*t*m42+s2+t2-u2+2*s*t-2*m2*m3*u+m32*(2*m42-2*s-2*t+u)+m22*(4*m32+2*m42-2*s-2*t+u))*m_X2+u*(m44-2*s*m42-2*t*m42+s2+t2-u2+2*s*t-2*m2*m3*u+m32*(2*m42-2*s-2*t+u)+m22*(4*m32+2*m42-2*s-2*t+u)))*m1+2*m_X4*((2*m32+m42-s-t)*m22+2*m3*(u-m42)*m2+s2+t2-m42*s+m32*(m42-s-t)-m42*t)-m42*(m42-u)*(m44-2*s*m42-2*t*m42+s2+t2-u2+2*s*t-2*m2*m3*u+m32*(2*m42-2*s-2*t+u)+m22*(4*m32+2*m42-2*s-2*t+u))-2*m_X2*m42*((4*m32-s-3*t+u)*m22+2*m3*(m42-u)*m2-(m42-s-t-u)*(s+t-u)+m32*(-3*s-t+u)))

    st = s_prop2*t_prop2*1/m_X4*2*(m_Gamma_X2+(m_X2-s)*(m_X2-t))*(2*m18+(4*m22+4*m32+4*m42-3*s-3*t-5*u)*m16+2*(-2*m23-2*(m3+m4)*m22+(-2*m32+s+u)*m2-2*m33-2*m32*m4+m4*(4*m_X2+u)+m3*(t+u))*m15+(2*m24+4*(m3+m4)*m23+(4*m32+4*m4*m3+4*m42-2*(s+t+3*u))*m22+(4*m33+4*m4*m32-2*u*m3-2*m4*(t+u))*m2+2*m34+4*m32*m42+4*u2+4*m33*m4-2*m32*s-3*m42*s-2*m3*m4*s-2*m32*t-3*m42*t+4*s*t-4*m_X2*(2*m22+2*m3*m2+2*m32-u)-6*m32*u-5*m42*u-2*m3*m4*u+4*s*u+4*t*u)*m14+(-4*m25-4*(m3+m4)*m24+4*(-2*m32-2*m42+s+t+2*u)*m23+(-8*m33-8*m4*m32+4*(-2*m42+s+t+2*u)*m3+2*m4*(s+3*(t+u)))*m22+(-4*m34+4*(-2*m42+s+t+2*u)*m32-s2+t2-3*u2-4*s*t-4*s*u-2*t*u+4*m42*(s+u))*m2-4*m35-8*m33*m42+m3*s2-m3*t2-3*m3*u2-2*m4*u2-4*m34*m4+4*m33*s+6*m32*m4*s+4*m33*t+4*m3*m42*t+2*m32*m4*t-4*m3*s*t+8*m33*u+4*m3*m42*u+6*m32*m4*u-2*m3*s*u-2*m4*s*u-4*m3*t*u-2*m4*t*u+4*m_X2*(2*m23+2*(m3+m4)*m22+(2*m32+s-t-u)*m2+2*m33+2*m32*m4-m3*(s-t+u)-2*m4*(s+t+u)))*m13+(4*(m3+m4)*m25+(4*m3*m4+s+t-u)*m24+(8*m33+8*m4*m32+(8*m42-4*(s+t+2*u))*m3-2*m4*(s+3*(t+u)))*m23+(8*m4*m33+(4*m42+3*(s+t-u))*m32-6*m4*(s+t+u)*m3+m42*(s+t-u)-2*(s2+t2-u2))*m22+(4*m35+4*m4*m34+(8*m42-4*(s+t+2*u))*m33-6*m4*(s+t+u)*m32-(4*u*m42+s2+t2-3*u2-2*t*u-2*s*(2*t+u))*m3+2*m4*(t2+2*u*t+u*(s+u)))*m2+s3+t3-u3-2*m32*s2-m42*s2+2*m3*m4*s2-2*m32*t2-m42*t2-s*t2+2*m32*u2+m42*u2+2*m3*m4*u2-s*u2-t*u2+4*m35*m4+m34*s+m32*m42*s-6*m33*m4*s+m34*t+m32*m42*t-s2*t-2*m33*m4*t+4*m42*s*t+8*m_X4*(m22+(m3-m4)*m2+m32-m3*m4-u)-m34*u-m32*m42*u+s2*u+t2*u-6*m33*m4*u+4*m3*m4*s*u+2*m3*m4*t*u-2*s*t*u-2*m_X2*(4*m24+(16*m32-8*m4*m3+12*m42-5*s-5*t-7*u)*m22-2*(4*m4*m32-4*m42*m3+2*u*m3+m4*s-m4*u)*m2+4*m34+m32*(12*m42-5*s-5*t-7*u)+2*m3*m4*(u-t)+2*u*(-3*m42+s+t+u)))*m12+(2*(2*m32-4*m4*m3-s+u)*m25+2*(2*m33-2*m4*m32-(s+t-u)*m3+m4*t)*m24+(4*m34-16*m4*m33+(-6*s-6*t+2*u)*m32+8*m4*(s+t+u)*m3+3*s2+t2-3*u2-2*t*u+2*m42*(-s+t+u))*m23+(4*m35-4*m4*m34+(-6*s-6*t+2*u)*m33+2*m4*(3*s+3*t+u)*m32-2*m42*(s+t-u)*m3+3*(s2+t2-u2)*m3-2*m4*t*(t+u))*m22+(-8*m4*m35-2*(s+t-u)*m34+8*m4*(s+t+u)*m33+(3*(s2+t2-u2)-2*m42*(s+t-u))*m32-2*m4*(s2+2*u*s+(t+u)**2)*m3-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(s2-4*t*s+t2-u2))*m2-8*m_X4*((m3-m4)*m22+(m32-4*m4*m3+m42-s)*m2-m32*m4+m3*(m42-t)+m4*u)+m3*(-2*(t-u)*m34+2*m4*s*m33+(2*(s-t+u)*m42+s2+3*t2-3*u2-2*s*u)*m32-2*m4*s*(s+u)*m3-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(s2-4*t*s+t2-u2))+4*m_X2*(m25+(m3-m4)*m24+2*(m32+m42-t-u)*m23+(2*m33-4*m4*m32+(6*m42-s-3*u)*m3-m4*(s+t-u))*m22+(m34+(6*m42-t-3*u)*m32-2*m4*(s+t)*m3-s2+t2+u2-s*t+m42*(s-2*u)+t*u)*m2+m35-m34*m4+2*m33*(m42-s-u)-m32*m4*(s+t-u)+m3*((t-2*u)*m42+s2-t2+u2+s*(u-t))+m4*(t*u+s*(2*t+u))))*m1-8*m_X4*((-m42+m3*m4+u)*m22+(m4*m32+(u-m42)*m3-m4*t)*m2-m3*m4*s+(m42-u)*u+m32*(u-m42))-m2*m3*(2*(2*m32-2*m4*m3-s+u)*m24+(2*m33-4*m4*m32-(s+t-u)*m3+2*m4*t)*m23+(4*m34-4*m4*m33+(-6*s-6*t+2*u)*m32+2*m4*(3*s+t+u)*m3+3*s2+t2-3*u2-2*t*u+2*m42*(-s+t+u))*m22-(4*m4*m34+(s+t-u)*m33-2*m4*(s+3*t+u)*m32+((s+t-u)*m42-s2-t2+u2)*m3+2*m4*t*(t+u))*m2-s3-t3+u3+m42*s2+m42*t2+s*t2-m42*u2+s*u2+t*u2+2*m33*m4*s+s2*t-4*m42*s*t-2*m34*(t-u)-s2*u-t2*u+2*s*t*u-2*m3*m4*s*(s+u)+m32*(2*(s-t+u)*m42+s2+3*t2-3*u2-2*s*u))+2*m_X2*(2*(m3-m4)*m25+(-2*m3*m4-s+t+u)*m24-2*(m3-m4)*(2*m3*m4+s+t+u)*m23-(4*m4*m33+(8*m42-6*u)*m32-2*m4*(s+2*t)*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m22+2*(m35-m4*m34+(2*m42-s-t-u)*m33+m4*(2*s+t)*m32+(-2*(s+t)*m42+2*s*t+s*u+t*u)*m3-m4*s*(t+u))*m2+m3*(-2*m4*m34+(s-t+u)*m33+2*m4*(s+t+u)*m32+((3*s-t+u)*m42-s2+t2-u2-2*s*u)*m3-2*m4*t*(s+u))))

    su = -s_prop2*u_prop2*(1/m_X4)*2*(m_Gamma_X2+(m_X2-s)*(m_X2-u))*(2*m18+(4*m22+4*m32+4*m42-3*s-5*t-3*u)*m16+2*(-2*m23-2*(m3+m4)*m22+(-2*m42+s+t)*m2-2*m43-2*m3*m42+4*m_X2*m3+m3*t+m4*t+m4*u)*m15+(2*m24+4*(m3+m4)*m23+(4*m32+4*m4*m3+4*m42-2*(s+3*t+u))*m22+(4*m43+4*m3*m42-2*t*m4-2*m3*(t+u))*m2+2*m44+4*m3*m43+4*m32*m42+4*t2-3*m32*s-2*m42*s-2*m3*m4*s-4*m_X2*(2*m22+2*m4*m2+2*m42-t)-5*m32*t-6*m42*t-2*m3*m4*t+4*s*t-3*m32*u-2*m42*u+4*s*u+4*t*u)*m14+(-4*m25-4*(m3+m4)*m24+4*(-2*m32-2*m42+s+2*t+u)*m23+(-8*m4*m32+2*(-4*m42+s+3*(t+u))*m3+4*m4*(-2*m42+s+2*t+u))*m22+(-4*m44+4*(s+2*t+u)*m42-s2-3*t2+u2-4*s*t+4*m32*(-2*m42+s+t)-4*s*u-2*t*u)*m2-4*m45-4*m3*m44-8*m32*m43+m4*s2-2*m3*t2-3*m4*t2-m4*u2+4*m43*s+6*m3*m42*s+8*m43*t+6*m3*m42*t+4*m32*m4*t-2*m3*s*t-2*m4*s*t+4*m43*u+2*m3*m42*u+4*m32*m4*u-4*m4*s*u-2*m3*t*u-4*m4*t*u+4*m_X2*(2*m23+2*(m3+m4)*m22+(2*m42+s-t-u)*m2+2*m3*(m42-s-t-u)+m4*(2*m42-s-t+u)))*m13+(4*(m3+m4)*m25+(4*m3*m4+s-t+u)*m24+(8*m4*m32+8*m42*m3-2*(s+3*(t+u))*m3+4*m4*(2*m42-s-2*t-u))*m23+(8*m3*m43+3*(s-t+u)*m42-6*m3*(s+t+u)*m4+m32*(4*m42+s-t+u)-2*(s2-t2+u2))*m22+((8*m43-4*m4*t)*m32+2*(2*m44-3*(s+t+u)*m42+(t+u)**2+s*t)*m3+m4*(4*m44-4*(s+2*t+u)*m42-s2+3*t2-u2+2*t*u+2*s*(t+2*u)))*m2+4*m3*m45+s3-t3+u3-m32*s2-2*m42*s2+2*m3*m4*s2+m32*t2+2*m42*t2+2*m3*m4*t2-s*t2-m32*u2-2*m42*u2-s*u2+t*u2+m44*s-6*m3*m43*s+m32*m42*s+8*m_X4*(m22+(m4-m3)*m2+m42-m3*m4-t)-m44*t-6*m3*m43*t-m32*m42*t+s2*t+4*m3*m4*s*t+m44*u-2*m3*m43*u+m32*m42*u-s2*u-t2*u+4*m32*s*u+2*m3*m4*t*u-2*s*t*u-2*m_X2*(4*m24+(12*m32-8*m4*m3+16*m42-5*s-7*t-5*u)*m22+2*(4*m4*m32+(-4*m42-s+t)*m3-2*m4*t)*m2+4*m44+2*t2-5*m42*s+6*m32*(2*m42-t)-7*m42*t+2*s*t+2*m3*m4*(t-u)-5*m42*u+2*t*u))*m12+(-2*(-2*m42+4*m3*m4+s-t)*m25-2*(m3*(2*m42-u)+m4*(-2*m42+s-t+u))*m24+(4*m44+(-6*s+2*t-6*u)*m42+8*m3*(-2*m42+s+t+u)*m4+3*s2-3*t2+u2-2*t*u+2*m32*(-s+t+u))*m23+(4*m45+(-6*s+2*t-6*u)*m43-2*m32*(s-t+u)*m4+3*(s2-t2+u2)*m4-2*m3*(2*m44-(3*s+t+3*u)*m42+u*(t+u)))*m22+(-2*(s-t+u)*m44+3*(s2-t2+u2)*m42-2*m3*(4*m44-4*(s+t+u)*m42+s2+(t+u)**2+2*s*t)*m4-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+s2*u+t2*u+2*s*t*u+m32*(-2*(s-t+u)*m42+s2-t2+u2-4*s*u))*m2+8*m_X4*((m3-m4)*m22-(m32-4*m4*m3+m42-s)*m2-m32*m4+m3*(m42-t)+m4*u)+m4*(2*(t-u)*m44+(s2-2*t*s-3*t2+3*u2)*m42+2*m3*s*(m42-s-t)*m4-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+s2*u+t2*u+2*s*t*u+m32*(2*(s+t-u)*m42+s2-t2+u2-4*s*u))+4*m_X2*(m25+(m4-m3)*m24+2*(m32+m42-t-u)*m23+(6*m4*m32-(4*m42+s-t+u)*m3+m4*(2*m42-s-3*t))*m22+(m44-(3*t+u)*m42-2*m3*(s+u)*m4-s2+t2+u2+m32*(6*m42+s-2*t)-s*u+t*u)*m2+m32*m4*(2*m42-2*t+u)+m4*(m44-2*(s+t)*m42+s2+t2-u2+s*(t-u))+m3*(-m44-(s-t+u)*m42+t*u+s*(t+2*u))))*m1+8*m_X4*((m32-m4*m3-t)*m22+(m4*m32+(u-m42)*m3-m4*t)*m2+m3*m4*s+m32*(m42-t)+t*(t-m42))-2*m_X2*(2*(m3-m4)*m25+(2*m3*m4+s-t-u)*m24-2*(m3-m4)*(2*m3*m4+s+t+u)*m23+(4*m3*m43-6*t*m42-2*m3*(s+2*u)*m4-s2+t2+u2+m32*(8*m42+s-t-3*u)+2*t*u)*m22-2*(2*m4*(m42-s-u)*m32-(m44-(2*s+u)*m42+s*(t+u))*m3+m4*(m44-(s+t+u)*m42+s*t+2*s*u+t*u))*m2+m4*(m4*(-3*s-t+u)*m32+2*(m42-s-t)*(m42-u)*m3-m4*(m42-s-t-u)*(s+t-u)))-m2*m4*(-2*(-2*m42+2*m3*m4+s-t)*m24+(m4*(2*m42-s+t-u)+m3*(2*u-4*m42))*m23+(4*m44+(-6*s+2*t-6*u)*m42+2*m3*(-2*m42+3*s+t+u)*m4+3*s2-3*t2+u2-2*t*u+2*m32*(-s+t+u))*m22-(m4*(s-t+u)*m32+2*(2*m44-(s+t+3*u)*m42+u*(t+u))*m3+m4*((s-t+u)*m42-s2+t2-u2))*m2-s3+t3-u3+m42*s2-3*m42*t2+s*t2+3*m42*u2+s*u2-t*u2+2*m3*m4*s*(m42-s-t)+2*m44*t-s2*t-2*m42*s*t-2*m44*u+s2*u+t2*u+2*s*t*u+m32*(2*(s+t-u)*m42+s2-t2+u2-4*s*u)))

    tu = t_prop2*u_prop2*(1/(m_X4))*2*(m_Gamma_X2+(m_X2-t)*(m_X2-u))*(2*m18+(4*m22+4*m32+4*m42-5*s-3*t-3*u)*m16+2*(-2*m33-2*m4*m32-2*m42*m3+s*m3+t*m3-2*m43+4*m_X2*m2+m4*s+m2*(-2*m32-2*m42+s)+m4*u)*m15+(-4*(2*m32+2*m4*m3+2*m42-s)*m_X2+m22*(4*m32+4*m42-5*s-3*t-3*u)+m2*(4*m33+4*m4*m32+4*m42*m3-2*(s+u)*m3+4*m43-2*m4*(s+t))+2*(m34+2*m4*m33+(2*m42-3*s-t-u)*m32+(2*m43-m4*s)*m3+m44+2*(s+t)*(s+u)-m42*(3*s+t+u)))*m14+(-4*m35-4*m4*m34-8*m42*m33+8*s*m33+4*t*m33+4*u*m33-8*m43*m32+8*m4*s*m32+4*m4*t*m32+4*m4*u*m32-4*m44*m3-3*s2*m3-t2*m3+u2*m3+8*m42*s*m3+4*m42*t*m3-4*s*t*m3+4*m42*u*m3-2*s*u*m3-4*t*u*m3-4*m45-3*m4*s2+m4*t2-m4*u2+8*m43*s+4*m43*t-2*m4*s*t+4*m43*u-4*m4*s*u-4*m4*t*u+4*m22*(-2*m33-2*m4*m32+(-2*m42+s+t)*m3+m4*(-2*m42+s+u))+4*m_X2*(2*m33+2*m4*m32+(2*m42-s+t-u)*m3+2*m2*(m32+m42-s-t-u)+m4*(2*m42-s-t+u))-2*m2*(2*m34+(4*m42-3*s-t-3*u)*m32+2*m44+s*(s+t+u)-m42*(3*s+3*t+u)))*m13+(4*m4*m35-s*m34+t*m34+u*m34+8*m43*m33-8*m4*s*m33-4*m4*t*m33-4*m4*u*m33+2*s2*m32-2*t2*m32-2*u2*m32-3*m42*s*m32+3*m42*t*m32+3*m42*u*m32+4*m45*m3+3*m4*s2*m3-m4*t2*m3-m4*u2*m3-8*m43*s*m3-4*m43*t*m3+2*m4*s*t*m3-4*m43*u*m3+2*m4*s*u*m3+4*m4*t*u*m3-s3+t3+u3+2*m42*s2-2*m42*t2+s*t2-2*m42*u2+s*u2-t*u2+8*m_X4*(m32+m4*m3+m42-m2*(m3+m4)-s)-m44*s+m44*t-s2*t+m44*u-s2*u-t2*u-2*s*t*u+m22*(8*m4*m33+(4*m42-s+t+u)*m32+(8*m43-4*m4*s)*m3+s2-t2-u2+4*t*u+m42*(-s+t+u))-2*m_X2*(4*m34+(16*m42-7*s-5*(t+u))*m32-4*m4*s*m3+4*m44+2*s2+2*m22*(6*m32+4*m4*m3+6*m42-3*s)-7*m42*s-5*m42*t+2*s*t-5*m42*u+2*s*u-2*m2*(4*m4*m32+(4*m42-s+t)*m3+m4*(u-s)))+2*m2*(2*m35+2*m4*m34+(4*m42-3*s-t-3*u)*m33+(4*m43-3*m4*(s+t+u))*m32+(2*m44-3*(s+t+u)*m42+s2+u2+s*t+2*s*u)*m3+m4*(2*m44-(3*s+3*t+u)*m42+s2+t2+2*s*t+s*u)))*m12+(4*m42*m35+2*s*m35-2*t*m35+4*m43*m34+2*m4*s*m34-2*m4*t*m34-2*m4*u*m34+4*m44*m33-3*s2*m33+3*t2*m33+u2*m33+2*m42*s*m33-6*m42*t*m33-6*m42*u*m33-2*s*u*m33+4*m45*m32-3*m4*s2*m32+3*m4*t2*m32+3*m4*u2*m32+2*m43*s*m32-6*m43*t*m32-6*m43*u*m32+s3*m3-t3*m3-u3*m3-3*m42*s2*m3+3*m42*t2*m3-s*t2*m3+3*m42*u2*m3-s*u2*m3+t*u2*m3+2*m44*s*m3-2*m44*t*m3+s2*t*m3-2*m44*u*m3+s2*u*m3+t2*u*m3+2*s*t*u*m3+m4*s3-m4*t3-m4*u3-3*m43*s2+m43*t2-m4*s*t2+3*m43*u2-m4*s*u2+m4*t*u2+2*m45*s+m4*s2*t-2*m43*s*t-2*m45*u+m4*s2*u+m4*t2*u+2*m4*s*t*u-8*m_X4*((m3+m4)*m22-(m32+4*m4*m3+m42-s)*m2+m32*m4+m3*(m42-t)-m4*u)+m22*(2*(s-t+u)*m33+2*m4*(s-t-u)*m32+(2*(s-t-u)*m42-s2+t2+u2-4*t*u)*m3+m4*(2*(s+t-u)*m42-s2+t2+u2-4*t*u))+4*m_X2*(m35+m4*m34+2*(m42-s-u)*m33+m4*(2*m42-3*s-t)*m32+(m44-(3*s+u)*m42+s2-t2+u2+s*u-t*u)*m3+m4*(m44-2*(s+t)*m42+s2+t2-u2+s*t-t*u)+m22*(2*m33+6*m4*m32+(6*m42-2*s+t)*m3+m4*(2*m42-2*s+u))-m2*(m34+(4*m42-s+t+u)*m32+2*m4*(t+u)*m3+m44-s*t-s*u-2*t*u+m42*(-s+t+u)))-2*m2*(4*m4*m35+(2*m42-u)*m34+4*m4*(2*m42-s-t-u)*m33+(2*m44-(s+3*(t+u))*m42+u*(s+u))*m32+m4*(4*m44-4*(s+t+u)*m42+s2+t2+u2+2*s*(t+u))*m3+m42*t*(-m42+s+t)))*m1+8*m_X4*((m32+m4*m3+m42-s)*m22+(-m4*m32+(u-m42)*m3+m4*t)*m2+s*(-m32-m4*m3-m42+s))+2*m_X2*(2*m4*m35+(s-t+u)*m34-2*m4*(s+t+u)*m33+(6*s*m42-s2+t2-u2-2*s*u)*m32+2*m4*(m44-(s+t+u)*m42+2*t*u+s*(t+u))*m3+m42*(m42-s-t-u)*(s+t-u)+m22*(4*m4*m33+(-8*m42+s-t+3*u)*m32+4*m4*(m42-t-u)*m3+m42*(s+3*t-u))-2*m2*(m35+m4*m34+(2*m42-s-t-u)*m33+m4*(2*m42-t-2*u)*m32+(m44-(2*t+u)*m42+t*(s+u))*m3+m4*(m42-s-t)*(m42-u)))-m3*m4*(2*(2*m42+s-t)*m34+m4*(2*m42+s-t-u)*m33+(4*m44+2*(s-3*(t+u))*m42-3*s2+3*t2+u2-2*s*u)*m32+m4*((s-t-u)*m42-s2+t2+u2)*m3+s3-t3-u3-3*m42*s2+m42*t2-s*t2+3*m42*u2-s*u2+t*u2+2*m44*s+s2*t-2*m42*s*t-2*m44*u+s2*u+t2*u+2*s*t*u+m22*(2*(s-t+u)*m32+m4*(s-t-u)*m3-s2+t2+u2+2*m42*(s+t-u)-4*t*u)-2*m2*(2*m4*m34+(2*m42-u)*m33+m4*(2*m42-s-3*t-u)*m32+(2*m44-(s+t+3*u)*m42+u*(s+u))*m3+m4*t*(-m42+s+t))))

    return vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_gen_ss(s, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2):
    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    ss = (((m1+m2)**2.)-s)*(((m3+m4)**2.)-s)*s_prop
    return 4.*vert*ss

@nb.jit(nopython=True, cache=True)
def M2_fi(s, t, m_d2, vert, m_X2, m_Gamma_X2):
    """
    aa --> dd
    """
    u = 3.*m_d2 - s - t
    s_prop2 = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_prop2 = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_prop2 = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)

    ss = 8*s_prop2*s_prop2*((m_X2-s)**2+m_Gamma_X2)*(2*(m_d2**2-2*m_d2*t+t*(s+t))+s**2)
    tt = (4*t_prop2*t_prop2*((m_X2-t)**2+m_Gamma_X2)*(2*m_X2*(m_d2**2-2*m_d2*(2*s+t)+2*s**2+2*s*t+t**2)+4*m_X2*m_d2**2*s+m_d2**2*(m_d2-t)**2))/m_X2
    uu = (4*u_prop2*u_prop2*(2*m_X2**2*(m_d2**2-2*m_d2*(s+t)+s**2+t**2)+4*m_X2*m_d2**2*s+m_d2**2*(-m_d2+s+t)**2)*((m_X2-2*m_d2+s+t)**2+m_Gamma_X2))/m_X2**2
    st = 2*s_prop2*4*t_prop2*(m_X2**2-m_X2*(s+t)+m_Gamma_X2+s*t)*(2*m_X2**2*(m_d2**2-m_d2*(s+2*t)+(s+t)**2)+m_X2*m_d2*(m_d2**2+m_d2*(s-2*t)+t**2))
    su = -2*s_prop2*4*u_prop2*(2*m_X2**2*(m_d2**2+m_d2*(s-2*t)+t**2)+m_X2*m_d2*(m_d2**2-m_d2*(s+2*t)+(s+t)**2))*((m_X2-s)*(m_X2-2*m_d2+s+t)+m_Gamma_X2)
    tu = -((4*t_prop2*u_prop2*(4*m_X2**2*s*(3*m_d2-s)+2*m_X2*m_d2*(2*m_d2**2-4*m_d2*(s+t)+s**2+2*s*t+2*t**2)+m_d2**4+m_d2**3*(s-2*t)+m_d2**2*t*(s+t))*((m_X2-t)*(m_X2-2*m_d2+s+t)+m_Gamma_X2))/m_X2**2)

    return vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_tr(s, t, m_d2, vert, m_X2, m_Gamma_X2):
    """
    ad --> dd
    """
    u = 3.*m_d2 - s - t
    s_prop2 = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_prop2 = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_prop2 = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)

    ss = 8*s_prop2*s_prop2*((m_X2-s)**2+m_Gamma_X2)*(2*m_d2**2-m_d2*(s+6*t)+s**2+2*s*t+2*t**2)
    tt = 8*t_prop2*t_prop2*((m_X2-t)**2+m_Gamma_X2)*(2*m_d2**2-m_d2*(6*s+t)+2*s**2+2*s*t+t**2)
    uu = 8*u_prop2*u_prop2*(8*m_d2**2-5*m_d2*(s+t)+s**2+t**2)*((m_X2-3*m_d2+s+t)**2+m_Gamma_X2)
    st = 16*s_prop2*t_prop2*(-2*m_d2+s+t)*(m_d2+s+t)*(m_X2**2-m_X2*(s+t)+m_Gamma_X2+s*t)
    su = -16*s_prop2*u_prop2*(t-4*m_d2)*(t-m_d2)*((m_X2-s)*(m_X2-3*m_d2+s+t)+m_Gamma_X2)
    tu = 16*t_prop2*u_prop2*(4*m_d2**2-5*m_d2*s+s**2)*((m_X2-t)*(m_X2-3*m_d2+s+t)+m_Gamma_X2)

    return vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_el(s, t, m_d2, vert, m_X2, m_Gamma_X2):
    """
    dd --> dd
    """

    u = 3.*m_d2 - s - t
    s_prop2 = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_prop2 = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_prop2 = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)
    
    ss = 8*s_prop2*s_prop2*((m_X2-s)**2+m_Gamma_X2)*(8*m_d2**2-8*m_d2*t+s**2+2*t*(s+t))
    tt = 8*t_prop2*t_prop2*((m_X2-t)**2+m_Gamma_X2)*(2*(s-2*m_d2)**2+2*s*t+t**2)
    uu = 8*u_prop2*u_prop2*(24*m_d2**2-8*m_d2*(s+t)+s**2+t**2)*((m_X2-4*m_d2+s+t)**2+m_Gamma_X2)
    st = 16*s_prop2*t_prop2*(-2*m_d2+s+t)*(2*m_d2+s+t)*(m_X2**2-m_X2*(s+t)+m_Gamma_X2+s*t)
    su = -16*s_prop2*u_prop2*(t-6*m_d2)*(t-2*m_d2)*((m_X2-s)*(m_X2-4*m_d2+s+t)+m_Gamma_X2)
    tu = 16*t_prop2*u_prop2*(12*m_d2**2-8*m_d2*s+s**2)*((m_X2-t)*(m_X2-4*m_d2+s+t)+m_Gamma_X2)

    return vert*(ss + tt + uu + st + su + tu)

# Cross-sections for each process el, fi, gen

@nb.jit(nopython=True, cache=True)
def ker_sigma_gen(t, s, p1cm, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub):
    return M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=sub)/(64.*np.pi*s*p1cm*p1cm)

# no factor taking care of identical particles (not known on this level)
def sigma_gen(s, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False):
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
    Heavysides from demanding positive p_cm^2: 
    H(1/(4*s)*[(s - m1/3 - m2/4)^2 - 4*m1/3^2*m2/4^2]) = H((s - m1/3 - m2/4)^2 - 4*m1/3^2*m2/4^2)
    = H(s - m1/3 - m2/4 - 2*m1/3*m2/4) = H(s - (m1/3 + m2/4)^2) = H(E_cm - m1/3 - m2/4)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(16*pi*[s^2 - 2*s(m1^2 + m2^2) + (m1^2 - m2^2)^2]) 
          * int_{t_lower}^{t_upper} dt |M|^2
    Note: This function can not be vectorized using 'quad' or 'quad_vec' as boundaries also will be arrays. 
          Use np.vectorize(sigma_gen)(s, m1, ...) instead if array output is wanted.
    """
    # Area where the three-momenta is defined, heavyside-functions - same as H(s-(m1+m2)^2)*H(s-(m3+m4)^2)
    if s < (m1 + m2)**2. or s < (m3 + m4)**2.:
        return 0.

    # Make upper and lower integration bounds 
    E1cm = (s + m1*m1 - m2*m2) / (2*np.sqrt(s))
    E3cm = (s + m3*m3 - m4*m4) / (2*np.sqrt(s))
    p1cm = np.sqrt((E1cm - m1)*(E1cm + m1))
    p3cm = np.sqrt((E3cm - m3)*(E3cm + m3))

    E13diff = (m1*m1 - m2*m2 - m3*m3 + m4*m4) / (2*np.sqrt(s))
    t_upper = (E13diff + (p1cm-p3cm))*(E13diff - (p1cm-p3cm))
    t_lower = (E13diff + (p1cm+p3cm))*(E13diff - (p1cm+p3cm))

    res, err = quad(ker_sigma_gen, t_lower, t_upper, args=(s, p1cm, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub), epsabs=0., epsrel=rtol_int)

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    m_d = 1e-5      # GeV 
    m_a = 0.
    m_X = 3*m_d
    sin2_2th = 1e-12
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    y = 2e-4

    # fi = aa->dd, tr = ad->dd, el = ss->ss
    vert_fi = y**4 * np.cos(th)**4*np.sin(th)**4
    vert_tr = y**4 * np.cos(th)**6*np.sin(th)**2
    vert_el = y**4 * np.cos(th)**8

    th_arr = np.linspace(0, 2*np.pi, 1000)
    Gamma = Gamma_X(y=y, th=th_arr, m_X=m_X, m_d=m_d)
    plt.plot(th_arr, Gamma)
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], labels=[r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$', r'$2\pi$'])
    plt.show()

    m1 = m_a
    m2 = m_d
    s_min = (m1 + m2)**2
    S = np.linspace(s_min, 100, int(1e3))
    T = np.linspace(-1, 1, int(1e3))
    s, t = np.meshgrid(S, T, indexing='ij')

    Gamma = Gamma_X(y=y, th=th, m_X=m_X, m_d=m_d)
    m_Gamma_X2 = (m_X*Gamma)**2
    print(Gamma)

    # sigma = np.vectorize(sigma_gen)(S, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2, sub=False)

    # plt.plot(S, sigma, 'r')
    # plt.show()

    # 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()

    M2_trans = M2_tr(s, t, m_d2=m_d**2, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    plot_M2 = ax1.contourf(s, t, np.log10(M2_trans), levels=300, cmap='jet')
    fig1.colorbar(plot_M2)
    ax1.set_xscale('log')

    # 2 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    M2_general = M2_gen(s, t, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    plot_M2 = ax2.contourf(s, t, np.log10(M2_general), levels=300, cmap='jet')
    cbar2 = fig2.colorbar(plot_M2)
    ax2.set_xscale('log')

    # Can make colorbar interactive - see constant value that you click
    from matplotlib import colors
    highlight_cmap = colors.ListedColormap(['k'])
    highlight = ax2.imshow(np.ma.masked_all_like(np.log(M2_general)), interpolation='nearest', vmin=np.log10(M2_general).min(), vmax=np.log10(M2_general).max(), extent=[S.min(),S.max(),T.min(), T.max()], cmap=highlight_cmap, origin='lower', aspect='auto', zorder=10)

    # highlight = [ax2.contour(s, t, (M2_general), colors='none')]

    def on_pick(event):
        val = event.mouseevent.ydata
        selection = np.ma.masked_outside(np.log10(M2_general), val-0.2, val+0.2)
        highlight.set_data(selection.T)
        # highlight[0].remove()
        # highlight[0] = ax2.contour(s, t, selection, colors='k')
        fig2.canvas.draw()
    cbar2.ax.set_picker(5)
    fig2.canvas.mpl_connect('pick_event', on_pick)

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()