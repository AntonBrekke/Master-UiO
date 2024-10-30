import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

a = -1
b = 1
V = b-a

N = int(1e7)
D = np.random.uniform(a, b, N)

f = lambda x: np.exp(-1/(2*0.1**2)*x**2)

x = np.linspace(a, b, int(1e5))
plt.plot(x, f(x))
std = 0.1
mean = 0.1
plt.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-1/(2*std**2)*(x - mean)**2))
plt.show()

# Standard Monte-Carlo
I = V*np.mean(f(D))
E = V*np.std(f(D)) / np.sqrt(N)

# Importance sampling (when weight function = uniform = 1/(b-a), get the above)
y = np.random.normal(mean, std, N)
weight_function = 1/(std*np.sqrt(2*np.pi)) * np.exp(-1/2*((y-mean)/std)**2)
g_weighted = f(y) / weight_function
IS = np.mean(g_weighted)
ES = np.std(g_weighted) / np.sqrt(N)

print(f'Integral I = {I:.8f}, error E = {E:.8f}, true value = {std*np.sqrt(2*np.pi):.8f}')
print(f'Integral I = {IS:.8f}, error E = {ES:.8f}, true value = {std*np.sqrt(2*np.pi):.8f}')