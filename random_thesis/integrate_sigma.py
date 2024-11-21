import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt 

"""
Bhaba-scattering in s >> m_e limit, 
constants omitted. 
"""

def dsigmadt(t, s): 
    return 1/s**2 * ((t/s)**2 + (s/t)**2 + (s+t)**2*(1/s + 1/t)**2)

def sigma_non_vec(s):
    return scint.quad(dsigmadt, -10, -1, args=s)[0]

def sigma(s):
    return np.vectorize(sigma_non_vec)(s)

s = np.linspace(1, 100, 1000)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

t_fixed = -10
ax1.loglog(s, dsigmadt(t=t_fixed, s=s), 'r', label=f'dsigma/dt, t={t_fixed}')
ax2.loglog(s, sigma(s), 'k', label='sigma')
ax1.grid(True)
ax2.grid(True)
ax1.legend()
ax2.legend()
plt.show()
