import numpy as np
import matplotlib.pyplot as plt 
import cmasher as cmr
from numba import njit
import time 

cmap = cmr.get_sub_cmap('Greys', 0, 1)

N = 1000
T = 2.5     # Unit k_B * K (or beta. Equivalent if k_B=1)
J = 1       # Coupling const

s_init = np.random.choice([-1, 1], size=(N,N))  # Randomize spin lattice, value -1: down, 1: up

@njit       # Using numba to speed up stuff 
def ising_model(s_init):
    s = s_init.copy()
    for iter in range(1, 100*N**2):
        # Choose random lattice point 
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        # Periodic boundary condition 
        top = s[N-1,j] if i == 1 else s[i-1,j]
        bottom = s[1,j] if i == N-1 else s[i+1,j]
        left = s[i,N-1] if j == 1 else s[i,j-1] 
        right = s[i,1] if j == N-1 else s[i,j+1]

        Ediff = 2*J*s[i,j]*(top + bottom + left + right)

        # Ediff = deltaU(i,j)
        # Metropolis algorithm using Boltzmann dist
        if Ediff <= 0: 
            s[i,j] = -s[i,j]
        else:
            k = np.random.uniform(0,1)
            if k < np.exp(-Ediff/T):
                s[i,j] = -s[i,j]
    return s 
"""
# Numba don't allow me to outsource this function 
@njit
def deltaU(i,j):
    # Periodic boundary condition 
    top = s[N-1,j] if i == 1 else s[i-1,j]
    bottom = s[1,j] if i == N-1 else s[i+1,j]
    left = s[i,N-1] if j == 1 else s[i,j-1] 
    right = s[i,1] if j == N-1 else s[i,j+1]

    Ediff = 2*s[i,j]*(top + bottom + left + right)
    return Ediff
"""

t0 = time.time()
s = ising_model(s_init)
t1 = time.time()
print(f'Runtime: {t1-t0}s')

fig = plt.figure()
ax = fig.add_subplot(122)
ax2 = fig.add_subplot(121)

spins = ax.imshow(s, cmap=cmap)
spins_init = ax2.imshow(s_init, cmap=cmap)

# cbar = fig.colorbar(spins, ax=ax, location='right', ticks=[-1, 0, 1])
# cbar.set_ticklabels(['-1', '0', '1'])
ax.invert_yaxis()
ax2.invert_yaxis()

fig.tight_layout()
plt.show()