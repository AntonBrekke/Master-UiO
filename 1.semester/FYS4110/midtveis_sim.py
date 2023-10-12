import numpy as np
import matplotlib.pyplot as plt 

"""
Task 2g) midterm FYS4110
Represent + measure as 1, - measure as -1 in code.
"""

def P(s, eig):
    # Probability for measuring either + or - 
    return 1/2 * (1 + s*np.imag(eig))      # P(\pm) = 1/2 * (1 \pm \sin(\theta))

def iterate_circuit(N_iter, eig0):
    eig = np.zeros(N_iter, dtype=complex)       # Contatining eigenvalue for each iteration 
    eig[0] = eig0 
    for n in range(0, N_iter-1):
        s = np.random.choice((-1,1), p=[P(-1, eig[n]), P(1, eig[n])])        # s is either + or -, s is \pm
        eig[n+1] = eig[n] * np.exp(1j*(-1*s * abs(ze)))        # -1*\pm = \mp (-1*+- = -+)
    return eig 


ze = 1e-3           # Smaller -> more iterations for convergence, less "noise"        
theta0 = np.pi/2    # initial angle for eigenvalue 
eig0 = np.exp(1j*theta0)        # Initial eigenvalue
N_iter = int(1e4)        

eig = iterate_circuit(N_iter, eig0)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title(r'$\theta = \pi,$' + r'$\quad|z||\epsilon|$ = ' + f'{ze:.0e}', fontsize=20)

ax.plot(range(0, N_iter), eig.real, 'tab:blue')
ax.plot(range(0, N_iter), eig.imag,'r')
ax.plot(range(0, N_iter), abs(eig),'g')
ax.set_xlabel('Number of iterations', fontsize=16)
ax.legend(['Re(eig)', 'Im(eig)'], prop={'size':16})

plt.show()