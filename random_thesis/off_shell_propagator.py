import numpy as np
import matplotlib.pyplot as plt 

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

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

def D_RIS2(s, M, Gamma, off_shell=True):
    if off_shell:
        # Off-shell propagator 
        propagator_RIS = ((s-M**2)**2 - (M*Gamma)**2)/((s-M**2)**2 + (M*Gamma)**2)**2
    if not off_shell:
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

# Propagator centered around M^2, with width ~ M*Gamma. Choose to plot around 10*width
s = np.linspace(M**2-10*M*Gamma, M**2+10*M*Gamma, int(1e4))
ds = s[1]-s[0]
# Integrate BW should give ~1
print(1/np.pi*M*Gamma*np.sum(D_BW2(s, M, Gamma))*ds)
# Integrate off-shell should give ~zero
print(1/np.pi*M*Gamma*np.sum(D_RIS2(s, M, Gamma, off_shell=True))*ds)
# Integrate on-shell should give ~1
print(1/np.pi*M*Gamma*np.sum(D_RIS2(s, M, Gamma, off_shell=False))*ds)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax2 = fig.add_subplot(122)
ax2.grid(True)

# Full BW propagator
ax1.plot(s, D_BW2(s, M, Gamma), color='tab:blue', lw=2, label=r'$D_{BW}$')
# Off-shell
ax2.plot(s, D_RIS2(s, M, Gamma, off_shell=True), color='r', lw=2, label=r'$D_{off}$')
# On-shell
ax2.plot(s, D_RIS2(s, M, Gamma, off_shell=False), color='green', lw=2, label=r'$D_{on}$')

# Width at half maximum
HM1 = 0.5 * np.max(D_BW2(s, M, Gamma))
HM2 = 0.5 * np.max(D_RIS2(s, M, Gamma, off_shell=False))
ax1.plot([M**2 - M*Gamma, M**2 + M*Gamma], [HM1, HM1], color='gray', ls='--')
ax2.plot([M**2 - M*Gamma, M**2 + M*Gamma], [HM2, HM2], color='gray', ls='--')

ax1.set_xlim(s[0], s[-1])
ax2.set_xlim(s[0], s[-1])

# Reduce number of ticks to make it more readable
ax1_xticks = ax1.get_xticks()
ax2_xticks = ax2.get_xticks()
ax1.set_xticks(ax1_xticks[::1])
ax2.set_xticks(ax2_xticks[::1])

ax1.legend(prop={'size':14})
ax2.legend(prop={'size':14})
fig.tight_layout()
plt.show()