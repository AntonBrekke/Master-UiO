import numpy as np
import matplotlib.pyplot as plt 

# Defining parameters of equation
k = 2
m = 1 
b = 1 
F0 = 1
w0 = 3

# Derived parameters 
w = np.sqrt(k/m)
gamma = b / (2*np.sqrt(m*k))
wp = -gamma*w + w*np.sqrt(gamma**2 - 1, dtype=complex)
wm = -gamma*w - w*np.sqrt(gamma**2 - 1, dtype=complex)

A = 1 / (1/(wp*wm) - wp/wm*(1/(wp - wm))) * (z0 - vz0 - F/m*(1 / ((1j*w0 - wp)*(1j*w0 - wm)))*(1 - 1j*w0/wm))
B = vz0 - w0/wm*(1j*F0/m * 1/((1j*w0 - wp)*(1j*w0 - wm))) - wp / wm * A/(wp - wm)

# Solution to the complex equation z'' + 2*gamma*w*z' + w^2*z = F0/m*exp(i*w0*t)
z = F0 / m * np.exp(1j*w0*t) / ((1j*w0 - wp)*(1j*w0 - wm)) + A / (wp - wm)*np.exp(wp*t) + B*np.exp(wm*t)

# The real solution is just x = Re(z) of x'' + 2*gamma*w*x' + w^2*x = F0/m*cos(w0*t)
x = np.real(z)