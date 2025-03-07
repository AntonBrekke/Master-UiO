import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import quad 
import numba as nb

@nb.njit
def M2_Dark_Higgs_Vector(t, m_X, m_phi, g):
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2

    m_phi2 = m_phi*m_phi

    fac = (2*m_phi2-m_X2+2*t)
    prop = 1/(t-m_phi2)**2

    M2 = g**4*(fac*(fac/prop - 2/m_X2) + prop/m_X4)
    return M2

@nb.njit
def sigma_hh_dd(s, m_X, m_phi, g):
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_phi2 = m_phi*m_phi
    m_phi4 = m_phi2*m_phi2
    s2 = s*s
    # Anton: Heavyside-functions
    if s < 4*m_phi2 or s < 4*m_X2:
        return 0. 

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_phi2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    Xt = 2*m_phi2 - m_X2
    Xu = 2*s - 6*m_phi2 - 3*m_X2
    Yt = m_phi2
    Yu = s - m_phi2 - 2*m_X2

    int_t_M2t_upper = g**4*((Xt-2*Yt)/(Yt-t_upper)-4*(Xt-2*Yt)*np.log(2*Yt-2*t_upper)+4*t_upper-2*Xt-2/m_X2*(t_upper+Xt)*t_upper+1/(3*m_X4)*(t_upper-Yt)**3)

    int_t_M2t_lower = g**4*((Xt-2*Yt)/(Yt-t_lower)-4*(Xt-2*Yt)*np.log(2*Yt-2*t_lower)+4*t_lower-2*Xt-2/m_X2*(t_lower+Xt)*t_lower+1/(3*m_X4)*(t_lower-Yt)**3)

    int_t_M2u_upper = g**4*((Xu-2*Yu)/(Yu-t_upper)-4*(Xu-2*Yu)*np.log(2*Yu-2*t_upper)+4*t_upper-2*Xu-2/m_X2*(t_upper+Xu)*t_upper+1/(3*m_X4)*(t_upper-Yu)**3)

    int_t_M2u_lower = g**4*((Xu-2*Yu)/(Yu-t_lower)-4*(Xu-2*Yu)*np.log(2*Yu-2*t_lower)+4*t_lower-2*Xu-2/m_X2*(t_lower+Xu)*t_lower+1/(3*m_X4)*(t_lower-Yu)**3)

    int_t_2REMtMu_upper = g**4*((s-4*m_phi2+m_X2)**2*(np.log(t_upper-m_phi2)-np.log(s+t_upper-m_phi2-2*m_X2))/(2*m_X2-s) - 2*(s-4*m_phi2+m_X2)*t_upper/m_X2 + 1/m_X4*(m_X2*(t_upper**2-2*m_phi2*t_upper)+m_phi2*(t_upper*(s+t_upper)-m_phi2*t_upper)-t_upper**2*(s/2+t_upper/3)))

    int_t_2REMtMu_lower = g**4*((s-4*m_phi2+m_X2)**2*(np.log(t_lower-m_phi2)-np.log(s+t_lower-m_phi2-2*m_X2))/(2*m_X2-s) - 2*(s-4*m_phi2+m_X2)*t_lower/m_X2 + 1/m_X4*(m_X2*(t_lower**2-2*m_phi2*t_lower)+m_phi2*(t_lower*(s+t_lower)-m_phi2*t_lower)-t_lower**2*(s/2+t_lower/3)))

    int_t_M2_upper = int_t_M2t_upper + int_t_M2u_upper + 2*int_t_2REMtMu_upper
    int_t_M2_lower = int_t_M2t_lower + int_t_M2u_lower + 2*int_t_2REMtMu_lower

    sigma = (int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm)

    return sigma / 2

@nb.njit
def sigma_hh_dd_new(s, m_X, m_phi, g):
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_phi2 = m_phi*m_phi
    m_phi4 = m_phi2*m_phi2
    s2 = s*s
    # Anton: Heavyside-functions
    if s < 4*m_phi2 or s < 4*m_X2:
        return 0. 

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_phi2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    Xt = 2*m_phi2 - m_X2
    Xu = 2*s - 6*m_phi2 - 3*m_X2
    Yt = m_phi2
    Yu = s - m_phi2 - 2*m_X2

    # Removed longitudinal contribution
    int_t_M2t_upper = g**4*((Xt-2*Yt)/(Yt-t_upper)-4*(Xt-2*Yt)*np.log(2*Yt-2*t_upper)+4*t_upper-2*Xt)

    int_t_M2t_lower = g**4*((Xt-2*Yt)/(Yt-t_lower)-4*(Xt-2*Yt)*np.log(2*Yt-2*t_lower)+4*t_lower-2*Xt)

    int_t_M2u_upper = g**4*((Xu-2*Yu)/(Yu-t_upper)-4*(Xu-2*Yu)*np.log(2*Yu-2*t_upper)+4*t_upper-2*Xu)

    int_t_M2u_lower = g**4*((Xu-2*Yu)/(Yu-t_lower)-4*(Xu-2*Yu)*np.log(2*Yu-2*t_lower)+4*t_lower-2*Xu)

    int_t_2REMtMu_upper = g**4*((s-4*m_phi2+m_X2)**2*(np.log(t_upper-m_phi2)-np.log(s+t_upper-m_phi2-2*m_X2))/(2*m_X2-s))

    int_t_2REMtMu_lower = g**4*((s-4*m_phi2+m_X2)**2*(np.log(t_lower-m_phi2)-np.log(s+t_lower-m_phi2-2*m_X2))/(2*m_X2-s))

    int_t_M2_upper = int_t_M2t_upper + int_t_M2u_upper + 2*int_t_2REMtMu_upper
    int_t_M2_lower = int_t_M2t_lower + int_t_M2u_lower + 2*int_t_2REMtMu_lower

    sigma = (int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm)

    return sigma / 2

@nb.njit
def sigma_hh_dd_real(s, m_X, m_d, m_phi, g):
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_d2 = m_d*m_d
    m_phi2 = m_phi*m_phi
    m_phi4 = m_phi2*m_phi2
    s2 = s*s
    # Anton: Heavyside-functions
    if s < 4*m_phi2 or s < 4*m_X2:
        return 0. 

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_phi2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j
    
    int_t_M2 = 64*m_d2*g**4/(s-m_phi2)**2*(4+(s-2*m_X2)**2/m_X4)*(s-4*m_d2)*(t_upper-t_lower)

    sigma = int_t_M2.real / (64.*np.pi*s*p1cm*p1cm)

    return sigma / 2

@nb.njit
def sigma_XX_dd(s, m_X, m_d, g):
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
    m_d4 = m_d2*m_d2
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4

    # Anton: Heavyside-functions
    if s < 4*m_d2 or s < 4*m_X2:
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
    int_t_M2_upper = 8*g**4*((m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper)+(4*m_d2*t_upper*(s-4*m_X2))/m_X4+((4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*(np.log(m_d2-t_upper)-np.log(m_d2+2*m_X2-s-t_upper)))/(2*m_X6-m_X4*s)-2*t_upper)

    int_t_M2_lower = 8*g**4*((m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower)+(4*m_d2*t_lower*(s-4*m_X2))/m_X4+((4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*(np.log(m_d2-t_lower)-np.log(m_d2+2*m_X2-s-t_lower)))/(2*m_X6-m_X4*s)-2*t_lower)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma / 2

@nb.njit
def sigma_XX_dd_new(s, m_d, m_X, vert):
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
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    s2 = s*s
    # Anton: Heavyside-functions
    if s < 4*m_d**2 or s < 4*m_X**2:
        return 0. 

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_d2)
    p3cm = np.sqrt(0.25*s - m_X2)

    E1cm = np.sqrt((p1cm-m_d)*(p1cm+m_d))

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    int_t_M2_upper = 8*vert*((m_X2-6*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)+(m_X2-6*m_d2)**2/(t_upper-m_d2)+((24*m_d4+m_d2*(12*s-40*m_X2)+4*m_X4+s2)*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper)))/(2*m_X2-s)-2*t_upper)

    int_t_M2_lower = 8*vert*((m_X2-6*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)+(m_X2-6*m_d2)**2/(t_lower-m_d2)+((24*m_d4+m_d2*(12*s-40*m_X2)+4*m_X4+s2)*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower)))/(2*m_X2-s)-2*t_lower)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma / 2

@nb.njit
def sigma_tot(s, m_X, m_d, m_h, g):
    m_d2 = m_d*m_d
    m_d3 = m_d*m_d2
    m_d4 = m_d2*m_d2

    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    m_h2 = m_h*m_h
    m_h4 = m_h2*m_h2

    s2 = s*s
    s3 = s*s2
    if s < 4*m_d2 or s < 4*m_X2:
        return 0. 
    
    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_d2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    int_t_M2_upper = 8*g**4*(-(1/(m_X4*(m_h2-s)*(s-2*m_X2)))*(m_h2*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))+4*m_d4*(-32*m_X6+32*m_X4*s-16*m_X2*s2+3*s3)+4*m_d2*(8*m_X8+3*m_X4*s2-m_X2*s3)-m_X4*s*(4*m_X4+s2))*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper))-(2*t_upper*(m_h4*(m_d2*(8*m_X2-2*s)+m_X4)+2*m_h2*(4*m_d2*(2*m_X4-3*m_X2*s+s2)-m_X4*s)+8*m_d4*(12*m_X4-4*m_X2*s+s2)-8*m_d2*s*(5*m_X4-3*m_X2*s+s2)+m_X4*s2))/(m_X4*(m_h2-s)**2)+(m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper))

    int_t_M2_lower = 8*g**4*(-(1/(m_X4*(m_h2-s)*(s-2*m_X2)))*(m_h2*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))+4*m_d4*(-32*m_X6+32*m_X4*s-16*m_X2*s2+3*s3)+4*m_d2*(8*m_X8+3*m_X4*s2-m_X2*s3)-m_X4*s*(4*m_X4+s2))*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower))-(2*t_lower*(m_h4*(m_d2*(8*m_X2-2*s)+m_X4)+2*m_h2*(4*m_d2*(2*m_X4-3*m_X2*s+s2)-m_X4*s)+8*m_d4*(12*m_X4-4*m_X2*s+s2)-8*m_d2*s*(5*m_X4-3*m_X2*s+s2)+m_X4*s2))/(m_X4*(m_h2-s)**2)+(m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower))

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    
    return sigma / 2

m_ratio = 3
m_d = 30e-6
m_X = m_ratio * m_d
m_phi = m_X
g = 1e-3
s = 10**(np.linspace(np.log10(4*m_X**4), 4, int(1e5)))

sigma_tot = np.vectorize(sigma_tot)(s, m_X, m_d, m_phi, g)
sigma_Vector = np.vectorize(sigma_XX_dd)(s, m_X, m_d, g)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.axvline(4*m_X**2, color='k', linestyle='--')
ax2.axvline(4*m_X**2, color='k', linestyle='--')

ax1.loglog(s, sigma_Vector, 'r')
ax2.loglog(s, sigma_tot, 'tab:green')

plt.show()
