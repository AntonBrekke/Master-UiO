import numpy as np
import matplotlib.pyplot as plt 

def viz_grid(X, Y):
    x,y = np.meshgrid(X, Y)
    plt.plot(x, y, color='k', marker='.', linestyle='None')


Nx = 10
Ny = 20
# X = np.concatenate([np.linspace(0, 0.5, 2*Nx), np.linspace(0.5, 1, Nx)])
X = np.linspace(0, 1, Nx)
Y = np.linspace(0, 1, Ny)

x, y = np.meshgrid(X, Y, indexing='ij')

f = 0.1*(x-2)**2 + y**2

viz_grid(X, Y)
plt.contour(x, y, f, levels=20)
plt.show()