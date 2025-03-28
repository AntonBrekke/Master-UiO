import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.special as scs

"""
Conclusion: 
Radau seems like the best for this case
max_step plays a huge role -- standard is max_step = np.inf.
Try max_step = np.inf vs. max_step = 1, max_step = 1e-2 etc. 
"""

a = 1
k = 100

def fprime(x, y):
    # RHS of the ODE 
    # return [np.exp(-a*x)*(k*np.cos(k*x)-a*np.sin(k*x))]
    # return [(1 - 2*(x>0))*x]
    return [np.sin(k*x)/(k*x)]

def event_zero(x, y):
    return y

x0 = 1
x1 = 5
Nx = 1e4
x = np.linspace(x0, x1, int(Nx))
y0 = np.array([1])          # y(x0) = y0

dx = x[1]-x[0]
atol = 0
rtol = 1e-3

max_step = 4e-2

sol_RK45 = solve_ivp(fun=fprime, t_span=(x0, x1), y0=y0, method='RK45', t_eval=x, events=[event_zero], atol=atol, rtol=rtol, max_step=max_step)
sol_Radau = solve_ivp(fun=fprime, t_span=(x0, x1), y0=y0, method='Radau', t_eval=x, events=[event_zero], atol=atol, rtol=rtol, max_step=max_step)
sol_LSODA = solve_ivp(fun=fprime, t_span=(x0, x1), y0=y0, method='LSODA', t_eval=x, events=[event_zero], atol=atol, rtol=rtol, max_step=max_step)

# sol_exact = np.sin(k*x)/(x) - np.sin(k*x0)/(x0) + y0
# sol_exact = np.exp(-a*np.cos(k1*x)*x)*np.sin(k*x) - np.exp(-a*np.cos(k1*x0)*x0)*np.sin(k*x0) + y0
# sol_exact = np.exp(-a*x)*np.sin(k*x) - np.exp(-a*x0)*np.sin(k*x0) + y0
sol_exact = 1/k*scs.sici(k*x)[0] - 1/k*scs.sici(k*x0)[0] + y0

plt.plot(x, sol_RK45.y[0], 'tab:blue', label='RK45')
plt.plot(x, sol_Radau.y[0], 'r', label='Radau')
plt.plot(x, sol_LSODA.y[0], 'tab:green', label='LSODA')
plt.plot(x, sol_exact, 'k', label='Analytic', ls=':')

print('MSE RK45:', 1/Nx*np.sum((sol_exact - sol_RK45.y[0])**2))
print('MSE Radau:', 1/Nx*np.sum((sol_exact - sol_Radau.y[0])**2))
print('MSE LSODA:', 1/Nx*np.sum((sol_exact - sol_LSODA.y[0])**2))

plt.grid(True)
plt.legend()
plt.show()