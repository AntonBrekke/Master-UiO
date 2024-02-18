import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import sympy.physics.mechanics as spm

m1, m2, l1, l2, l3, l4, g = sy.symbols('m1 m2 l1 l2 l3 l4 g')

q1, q2, q3 = spm.dynamicsymbols('q1 q2 q3')
q1d, q2d, q3d = spm.dynamicsymbols('q1 q2 q3', 1)       # First order derivative of q1, q2, q3
q1dd, q2dd, q3dd = spm.dynamicsymbols('q1 q2 q3', 2)       # Second order derivative of q1, q2, q3

# Implement Lagrangian of the system and get the equations of motion
K = 1/2*m1*(l1**2*q1d**2 + l3**2*q2d**2 + 2*l1*l3*q1d*q2d*sy.cos(q1 - q2)) + 1/2*m2*(l2**2*q1d**2 + l4**2*q3d**2 + 2*l2*l4*q1d*q3d*sy.sin(q1 + q3))
V = m1*g*(-l1*sy.cos(q1) - l3*sy.cos(q2)) + m2*g*(l2*sy.cos(q1) - l4*sy.sin(q3))
L = K - V

LM = spm.LagrangesMethod(L, [q1, q2, q3])
eom_theta, eom_phi, eom_psi = LM.form_lagranges_equations()

sol = sy.solve([eom_theta, eom_phi, eom_psi], [q1dd, q2dd, q3dd])

x2d_sol = sol[q1dd]
x4d_sol = sol[q2dd]
x6d_sol = sol[q3dd]

print(sy.latex(x6d_sol))
print('')
print(sy.latex(sy.simplify(x6d_sol)))       # Make sympy try to simplify eq.
