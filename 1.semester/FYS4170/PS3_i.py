import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.seterr(divide='ignore')

t_tot = 500
t_i = 0
Nt = 1000
Nz = 10000

z = np.linspace(0, t_tot, Nz)
t = np.linspace(t_i, t_tot, Nt)

Z, T = np.meshgrid(z, t, indexing='ij')

dz = 5
dp = 1 / dz
p0 = 1
m = 1

def wave(z, t, dp=1, p0=1):
    a = -1 / (2*dp**2)
    b = -1j / (2*m)*t
    c = 1j*z
    n = -1*(a+b)
    l = -1*(2*a*p0 - c)
    return np.sqrt(np.pi/n) * np.exp(a*p0**2 + l**2/(4*n) - 1j*m*t)

Psi = wave(Z, T, dp, p0)
Psi_real = Psi.real
Psi_imag = Psi.imag
Psi_sq = abs(Psi)**2
Psi_env = abs(Psi)

fig = plt.figure()
ax = fig.add_subplot()

real_wave, = ax.plot(z, Psi_real[:,0], 'r')
complex_wave, = ax.plot(z, Psi_imag[:,0], 'tab:blue')
sq_wave, = ax.plot(z, Psi_sq[:,0], 'tab:green')
env_wave_p, = ax.plot(z, Psi_sq[:,0], 'k', alpha=0.5)
env_wave_n, = ax.plot(z, Psi_sq[:,0], 'k', alpha=0.5)
t_center = 450
time = ax.text(z[int(t_center * Nz / t_tot)], 0.5, f't={t[-1]:.2f}s', color='k', ha='center', va='top')

def update(frame):
    real_wave.set_data(z, Psi_real[:,frame])
    complex_wave.set_data(z, Psi_imag[:,frame])
    sq_wave.set_data(z, Psi_sq[:,frame])
    env_wave_p.set_data(z, Psi_env[:,frame])
    env_wave_n.set_data(z, -Psi_env[:,frame])
    time.set_text(f't = {t[frame]:.2f}s')
    return real_wave, complex_wave, sq_wave, env_wave_p, env_wave_n, time

anim_speed = 1
ani = FuncAnimation(fig, update, frames=[i for i in range(0, len(t), anim_speed)],
                    blit=True, interval=5, repeat=True)
plt.show()

fig = plt.figure()
gs = fig.add_gridspec(3,1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

axes = np.array([ax1, ax2, ax3])

def plot_wave(i, t):
    dt = (t_tot - t_i) / Nt         # equally spaced
    t_index = int(t / dt)        # Since t = t_index * dt, dt = (b-a)/N
    axes[i].plot(z, Psi_real[:,t_index], 'r')
    axes[i].plot(z, Psi_imag[:,t_index], 'tab:blue')
    axes[i].plot(z, Psi_sq[:,t_index], 'tab:green')
    axes[i].plot(z, Psi_sq[:,t_index], 'k', alpha=0.5)
    axes[i].plot(z, Psi_sq[:,t_index], 'k', alpha=0.5)
    axes[i].text(450, 0.1, f't={t:.2f}s', color='k', ha='center', va='top')

plot_wave(0, 50)
plot_wave(1, 150)
plot_wave(2, 300)

fig.tight_layout()
plt.show()