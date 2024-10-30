import numpy as np
import matplotlib.pyplot as plt 

def EulerCromer(a=lambda t, v, x: 0, v0=0, x0=0, num_points=1e5, T=1):
    # Make your time-array
    N = int(num_points)
    t = np.linspace(0, T, N)
    dt = t[1] - t[0]

    # Make placeholders for v and x
    v = np.zeros(N)
    x = np.zeros(N)

    # Set initial conditions 
    v[0] = v0
    x[0] = x0

    # Integrate numerically! 
    for i in range(N-1):
        v[i+1] = v[i] + a(t[i], v[i], x[i])*dt
        x[i+1] = x[i] + v[i+1]*dt       # "Cromer"-step (i+1) in Euler-Cromer

    return t, v, x

# Any acceleration you want! :) 
k = 100
D = 1
dvdt = lambda t, v, x: -k*x - D*v

t, v, x = EulerCromer(a=dvdt, v0=10, x0=0, num_points=1e5, T=10)

# Present the results with subplots! 
fig = plt.figure(figsize=(8,5))
axes = fig.subplots(nrows=3, ncols=1, sharex=True)

axes[0].plot(t, x, 'r', label='x(t)')
axes[1].plot(t, v, 'tab:blue', label='v(t)')
axes[2].plot(t, dvdt(t,v,x), 'tab:green', label='a(t)')
[ax.grid(True) for ax in axes]
[ax.legend(fontsize=12) for ax in axes]
fig.tight_layout()
plt.show()