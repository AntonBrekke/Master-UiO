import numpy as np
import matplotlib.pyplot as plt

n = 20
p0 = 5
wavelength = 2*np.pi / p0
z = np.linspace(-n*wavelength, n*wavelength, 10000)[:,None]
dp = np.linspace(1e-2, 1e-1, 3)[None,:]

def Psi(z, dp):
    return dp * np.sqrt(2/np.pi) * np.exp(1j*p0*z) * np.exp(-2*dp**2 * z**2)

plt.plot(z, np.real(Psi(z,dp)), alpha=0.8)
plt.grid(True)
plt.xlabel('z', fontsize=16)
plt.legend([f'$\Delta$p={i:.2f}' for i in dp[0]], prop={'size':12})
plt.show()
