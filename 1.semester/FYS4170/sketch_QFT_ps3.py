import numpy as np
import matplotlib.pyplot as plt

n = 2
z = np.linspace(-n*np.pi, n*np.pi, 10000)[:,None]
a = np.linspace(0.1, 1, 3)[None,:]
p0 = 5

def Psi(z, a):
    return np.sqrt(a/np.pi)*np.cos(p0*z)*np.exp(-a*z**2)

plt.plot(z, Psi(z,a), '--')
plt.grid(True)
plt.legend([f'a={i}' for i in a[0]])
plt.show()
