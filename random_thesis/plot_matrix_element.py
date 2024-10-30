import numpy as np
import matplotlib.pyplot as plt 

m_s = 1
m_phi = 3*m_s

m_s2 = m_s**2
m_phi2 = m_phi**2
m_s4 = m_s2**2
m_phi4 = m_phi2**2

S = np.linspace(0, 1e1, int(1e3))
T = np.linspace(0, 1e1, int(1e3))

s, t = np.meshgrid(S, T, indexing='ij')

u = 2*m_s2 + 2*m_phi2 - s - t

M2_Depta = (s + 2*t - 2*m_phi2 - 2*m_s2)**2*(-m_s4 - m_phi4 - m_s2*(2*m_phi2 - s - 2*t) + 2*m_phi2*t - t*(s + t))
M2_Depta_v2 = (u-t)**2*((m_s2 + m_phi2 - u)*(m_s2 + m_phi2 - t) - m_phi2*s)
correction = 8*m_s2*((s-2*m_phi2)*((u - m_s2)*(t + m_s2 - m_phi2) + (t - m_s2)*(u + m_s2 - m_phi2)))
M2 = M2_Depta + correction

print(f'Max M2_value: {np.max(M2_Depta), np.max(M2)}')

def call_fig(proj_3D=False):
    fig = plt.figure(figsize=(9,5))
    ax1 = fig.add_subplot(121, projection='3d' if proj_3D is True else None)
    ax2 = fig.add_subplot(122, projection='3d' if proj_3D is True else None)

    plot1 = ax1.contourf(s, t, M2_Depta, levels=30)
    plot2 = ax2.contourf(s, t, M2, levels=30)
    fig.colorbar(plot1, )
    fig.colorbar(plot2)

    fig.tight_layout()
    plt.show()

call_fig(proj_3D=False)