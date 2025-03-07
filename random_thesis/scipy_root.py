import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import root 

def poly(x):
    # No real roots 
    return 2*x**3 + 3*x**2 - 5*x - 3


root_sol = root(fun=poly, x0=[-5, 5, 0], method='lm', tol=0)
print(root_sol.x)
print(poly(root_sol.x))