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

N = 1000            # Number of spins on each lattice side
T = 2               # Working with k_B = 1, thus T [J/k_B]
J = 1               # Coupling constant
M = 100*N**2        # Number of iterations in Monte-Carlo cycle. Each spin flipped M/N^2 times on average. 

s_init = np.random.choice([-1, 1], size=(N,N))  # Randomize spin lattice, value -1: down, 1: up
# s_init = 1*np.ones((N,N))


@njit       # Using numba to speed up function 
def ising_model(s_init):
    s = s_init.copy()
    C = np.zeros(M)
    for iter in range(0, M):
        # Choose random lattice point from 0 to N-1 (Monte Carlo)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        # Periodic boundary conditions 
        top = s[N-1,j] if i == 0 else s[i-1,j]
        bottom = s[0,j] if i == N-1 else s[i+1,j]
        left = s[i,N-1] if j == 0 else s[i,j-1] 
        right = s[i,0] if j == N-1 else s[i,j+1]

        """
        Energy: E = -J*sum_ij s_i*s_j (no external field)
        For some i, sum over j's. E_i = -J*s_i * sum_j s_j, j neighbour. 
        Flip s_i -> -s_i
        dE_i = after flip - before flip = 2*J*s_i * sum_j s_j 
        """
        # Difference in energy upon flipping spin 
        Ediff = 2*J*s[i,j]*(top + bottom + left + right)

        """
        Metropolis-Hastings:
        Have (spin)state x, propose state x' with g(x'|x) (here uniform dist.)
        Detailed balance: P(x|x')*P(x) = P(x'|x)*P(x') (prob. for x -> x' = prob. for x' -> x)
                       => P(x'|x)/P(x|x') = P(x)/P(x')
        Then prob. for transition P(x'|x) = g(x'|x)*A(x',x)
        A(x',x) prob. to accept new state x'. Then by insertion,
        g(x'|x)*A(x',x) / [g(x|x')*A(x,x')] = P(x)/P(x')        (**)
        Must find A(x',x) which satisfies (**) above. A (maybe non-trivial) suggestion is 
        A(x',x) = min{1, P(x')/P(x)*g(x|x')/g(x'|x)}, since either 
        A(x',x) or A(x,x') must be 1. 
        If g(x'|x) = g(x|x') symmetric (uniform, gaussian etc.)
        A(x',x) = min{1, P(x')/P(x)}
        P(E) = 1/Z * exp(-E/T). Then P(E')/P(E) = exp(-Ediff/T)
        """
        # Metropolis algorithm using Boltzmann dist.
        if Ediff <= 0:      # e^(-Ediff/T) >= 1 => A = 1 -> 100% accept 
            s[i,j] = -s[i,j]
        else:   # Don't want to get stuck on most probable states, but to cover state-space. Give chance to reject or accept. 
            r = np.random.uniform(0,1)
            if r < np.exp(-Ediff/T):
                s[i,j] = -s[i,j]

    return s

print('Start running')
t0 = time.time()
s, C = ising_model(s_init)
t1 = time.time()
print(f'Runtime: {t1-t0:.3f}s')
print(f'N-cycles: {M:.1e}')

fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)

spins_init = ax1.imshow(s_init, cmap=cmap)
spins = ax2.imshow(s, cmap=cmap)

# cbar = fig.colorbar(spins, ax=ax, location='right', ticks=[-1, 0, 1])
# cbar.set_ticklabels(['-1', '0', '1'])
ax1.invert_yaxis()
ax2.invert_yaxis()

fig.tight_layout()
plt.show()
