import numpy as np
import matplotlib.pyplot as plt 
from numba import njit
import time 
import cmasher as cmr

"""
Simple code for solving Ising model using Monte Carlo.
Not optimized in terms of efficiency, and focus more on 
readability. Not suited for all the purposes of FYS3150 
done in C++. 
"""

cmap = cmr.get_sub_cmap('Greys', 0, 1)
plt.style.use('dark_background')

N = 1000  
T = 1    # Working with k_B = 1, thus T [J/k_B]
J = 1       # Coupling constant
M = 100*N**2        # Number of iterations in Monte-Carlo cycle 

s_init = np.random.choice([-1, 1], size=(N,N))  # Randomize spin lattice, value -1: down, 1: up
# s_init = 1*np.ones((N,N))


@njit       # Using numba to speed up stuff 
def ising_model(s_init):
    s = s_init.copy()
    for iter in range(1, M):
        # Choose random lattice point (Monte Carlo)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        # Periodic boundary condition 
        top = s[N-1,j] if i == 1 else s[i-1,j]
        bottom = s[1,j] if i == N-1 else s[i+1,j]
        left = s[i,N-1] if j == 1 else s[i,j-1] 
        right = s[i,1] if j == N-1 else s[i,j+1]

        # Difference in energy upon flipping spin 
        Ediff = 2*J*s[i,j]*(top + bottom + left + right)

        # Metropolis algorithm using Boltzmann dist.
        if Ediff <= 0: 
            s[i,j] = -s[i,j]
        else:
            r = np.random.uniform(0,1)
            if r < np.exp(-Ediff/T):
                s[i,j] = -s[i,j]

    return s

print('Start running')
t0 = time.time()
s = ising_model(s_init)
t1 = time.time()
print(f'Runtime: {t1-t0}s')
print(f'N-cycles: {M}')

fig = plt.figure()
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

spins_init = ax.imshow(s_init, cmap=cmap)
spins = ax2.imshow(s, cmap=cmap)

# cbar = fig.colorbar(spins, ax=ax, location='right', ticks=[-1, 0, 1])
# cbar.set_ticklabels(['-1', '0', '1'])
ax.invert_yaxis()
ax2.invert_yaxis()

fig.tight_layout()
plt.show()