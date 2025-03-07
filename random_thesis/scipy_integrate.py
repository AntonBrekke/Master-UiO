import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def fprime(x, y):
    # RHS of the ODE 
    return [np.sin(10*x)/(10*x)]

def event_zero(x, y):
    return y

x0 = 1
x1 = 10
Nx = 1e4
x = np.linspace(x0, x1, int(Nx))
y0 = np.array([1])          # y(x0) = y0

atol = 0
rtol = 1e-3

sol_RK45 = solve_ivp(fun=fprime, t_span=(x0, x1), y0=y0, method='RK45', t_eval=x, events=[event_zero], max_step=1, atol=atol, rtol=rtol)
sol_Radau = solve_ivp(fun=fprime, t_span=(x0, x1), y0=y0, method='Radau', t_eval=x, events=[event_zero], max_step=1, atol=atol, rtol=rtol)
sol_LSODA = solve_ivp(fun=fprime, t_span=(x0, x1), y0=y0, method='LSODA', t_eval=x, events=[event_zero], max_step=1, atol=atol, rtol=rtol)

plt.plot(x, sol_RK45.y.T, 'tab:blue')
plt.plot(x, sol_Radau.y.T, 'r')
plt.plot(x, sol_LSODA.y.T, 'tab:green')

plt.show()