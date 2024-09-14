import numpy as np
import matplotlib.pyplot as plt 

P = np.linspace(0, 2*np.pi, 30)
T = np.linspace(0, np.pi, 30)
P2 = np.linspace(0, 2*np.pi - np.pi/4, 30)
T2 = np.linspace(0, np.pi, 30)

t, p = np.meshgrid(T, P)
t2, p2 = np.meshgrid(T2, P2)

x = np.cos(p)*np.sin(t)
y = np.sin(p)*np.sin(t)
z = np.cos(t)

x2 = np.cos(p2)*np.sin(t2)
y2 = np.sin(p2)*np.sin(t2)
z2 = np.cos(t2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z, alpha=0.5)
ax.plot_surface(x2, y2, z2, color='r', alpha=0.5)

plt.show()