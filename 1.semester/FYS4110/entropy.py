import numpy as np
import matplotlib.pyplot as plt 

b0 = 1/4
gamma = np.linspace(0.1, 0.5, 4)[None,:]
p0 = 0.1
t0 = 0
T = 10
Nt = int(1e4)
t = np.linspace(t0, T, Nt)[:,None]

def bloch_vector(t):
    return np.sqrt(4*b0**2*np.exp(-4*gamma*t) + p0**2)

def S(r):
    pp = 1/2 * (1 + r)
    pm = 1/2 * (1 - r)
    return np.log(pp**(-pp) + np.log(pm**(-pm)))


r = bloch_vector(t)
S = S(r)

plt.plot(t, S)
plt.show()