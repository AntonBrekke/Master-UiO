import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.seterr(divide='ignore')

T = 10
z = np.linspace(0, T, 1000)
t = np.linspace(0, T, 1000)

def Psi(z,t):
    return 1 / (2*np.sqrt(np.pi)) * 1 / np.sqrt(abs(z-t))

fig = plt.figure()
ax = fig.add_subplot()

wave, = ax.plot(z, Psi(z,0), 'r')
t_center = 9
time = ax.text(z[int(t_center/T*len(t))], 0.5, f't={t[-1]:.2f}s', color='k', ha='center', va='top')
def update(frame):
    wave.set_data(z, Psi(z, t[frame]))
    time.set_text(f't = {t[frame]:.2f}s')
    return wave, time

anim_speed = 5
ani = FuncAnimation(fig, update, frames=[i for i in range(0, len(t), anim_speed)],
                    blit=True, interval=5, repeat=True)
plt.show()
