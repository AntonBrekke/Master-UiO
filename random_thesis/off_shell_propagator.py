import numpy as np
import matplotlib.pyplot as plt 

"""
In real intermediate state subtraction, we subtract
the on-shell contribution to the Breit-Wigner propagator
|D_BW|^2 --> pi/(M*Gamma) * delta(s - M**2)
by decomposing 
|D_BW|^2 = |D_on|^2 + |D_off|^2
where it is expected that 
|D_on|^2 --> pi/(M*Gamma) * delta(s - M**2)
|D_off|^2 --> 0
in the sense of distributions i.e. when integrated over. 
This is what we check. 
"""

M = 1
Gamma = 1e-5

def D_BW2(s, M, Gamma):
    propagator_BW = 1/((s-M**2)**2 + (M*Gamma)**2)
    return propagator_BW

def D_RIS2(s, M, Gamma, type=0):
    if type == 0:
        # Off-shell propagator 
        propagator_RIS = ((s-M**2)**2 - (M*Gamma)**2)/((s-M**2)**2 + (M*Gamma)**2)**2
    if type == 1:
        # On-shell propagator
        propagator_RIS = (2*(M*Gamma)**2)/((s-M**2)**2 + (M*Gamma)**2)**2
    return propagator_RIS

def delta(x, a, eps):
    # pi times delta-function
    repr1 = eps / ((x-a)**2 + eps**2)
    repr2 = 2*eps**3 / ((x-a)**2 + eps**2)**2
    return repr1


# x = np.linspace(-3, 3, int(1e5))
# dx = x[1]-x[0]
# a = 1
# # Integrate over delta should give 1 
# print(1/np.pi*np.sum(delta(x, a, 0.001))*dx)
# plt.plot(x, delta(x, a, 0.1), label='0.1')
# plt.plot(x, delta(x, a, 0.01), label='0.01')
# plt.plot(x, delta(x, a, 0.001), label='0.001')
# plt.legend()
# plt.show()

s = np.linspace(0.9999, 1.0001, int(1e6))
ds = s[1]-s[0]
# Integrate BW should give 1
print(1/np.pi*M*Gamma*np.sum(D_BW2(s, M, Gamma))*ds)
# Integrate off-shell should give zero
print(1/np.pi*M*Gamma*np.sum(D_RIS2(s, M, Gamma, type=0))*ds)
# Integrate on-shell should give 1
print(1/np.pi*M*Gamma*np.sum(D_RIS2(s, M, Gamma, type=1))*ds)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(s, D_BW2(s, M, Gamma))
# Off-shell
ax2.plot(s, D_RIS2(s, M, Gamma, type=0), color='r')
# On-shell
ax2.plot(s, D_RIS2(s, M, Gamma, type=1), color='green')

fig.tight_layout()
plt.show()