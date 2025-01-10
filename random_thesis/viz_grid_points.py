import numpy as np
import matplotlib.pyplot as plt 

def viz_grid(Nx, Ny):

    X = np.linspace(0, 1, int(Nx))
    Y = np.linspace(0, 1, int(Ny))
    x,y = np.meshgrid(X, Y)
    plt.plot(x, y, color='k', marker='.', linestyle='None')
    plt.show()

viz_grid(Nx=20, Ny=100)