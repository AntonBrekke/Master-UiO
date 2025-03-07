import numpy as np
import matplotlib.pyplot as plt 

def viz_grid(X, Y):
    x,y = np.meshgrid(X, Y)
    plt.plot(x, y, color='k', marker='.', linestyle='None')


Nx = 40
Ny = 40
# X = np.concatenate([np.linspace(0, 0.5, 2*Nx), np.linspace(0.5, 1, Nx)])
X = np.linspace(0, 1, Nx)
Y = np.linspace(0, 1, Ny)

X2 = np.linspace(1, 2, Nx)
Y2 = np.linspace(1, 2, Ny)

x, y = np.meshgrid(X, Y, indexing='ij')
x2, y2 = np.meshgrid(X2, Y2, indexing='ij')

f = 0.1*(x-2)**2 + y**2
f2 = 0.1*(x2-2)**2 + y2**2

viz_grid(X, Y)
# viz_grid(X2, Y2)
plt.contour(x, y, f, levels=20)
# plt.contour(x2, y2, f2, levels=20)
plt.show()