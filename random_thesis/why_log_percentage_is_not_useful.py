import numpy as np
import matplotlib.pyplot as plt 

"""
When you have logarithmically spaced arrays, the naive way to 
get the percentage of the progression is very misleading.
It should be fixed by using the logged values. 
percentage_bad = (x_i-x_0)/(x_f-x_0)
percentage_good = (log(x_i) - log(x_0)) / (log(x_f) - log(x_0)) = log(x_i/x_0)/log(x_f/x_0)
All the progression ends up in the last intervals.
Gets worse the larger the span of data. 
"""

n_m = 20         # 21
n_th = 100        # 81
# Anton: Search for keV-scale sterile neutrinos. Input is in GeV, so (m_d keV) = (1e-6*m_d GeV)
m_d_grid = 1e-6*np.logspace(0, 2.5, n_m)    # 10^a keV - 10^b keV, a=0, b=2.5
sin2_2th_grid = np.logspace(-18, -8, n_th)

md_log_p = (m_d_grid - m_d_grid[0])/(m_d_grid[-1] - m_d_grid[0])
sin2_2th_log_p = (sin2_2th_grid - sin2_2th_grid[0])/(sin2_2th_grid[-1] - sin2_2th_grid[0])

# Given logarithmic percentage, find corresponding linear percentage
md_log_p_current = 74.9
sin2_2th_log_p_current = 4.2e-6
# Get indices
md_diff = abs(md_log_p_current/100 - md_log_p)
md_current_idx = np.where(md_diff==np.min(md_diff))
sin2_2th_diff = abs(sin2_2th_log_p_current/100 - sin2_2th_log_p)
sin2_2th_current_idx = np.where(sin2_2th_diff==np.min(sin2_2th_diff))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# Bad representation of percentage  
ax1.plot(m_d_grid, md_log_p, 'r', label='log')
ax2.plot(sin2_2th_grid, sin2_2th_log_p, 'r', label='log')

# How to get back useful percentage 
md_log_p = np.log10(m_d_grid/m_d_grid[0])/np.log10(m_d_grid[-1]/m_d_grid[0])
sin2_2th_log_p = np.log10(sin2_2th_grid/sin2_2th_grid[0])/np.log10(sin2_2th_grid[-1]/sin2_2th_grid[0])
ax1.plot(m_d_grid, md_log_p, 'tab:blue', label='linear')
ax2.plot(sin2_2th_grid, sin2_2th_log_p, 'tab:blue', label='linear')

# Use indices found above in linear repr. 
print(f'Log percent: {md_log_p_current:.2f}%, Linear percent: {100*md_log_p[md_current_idx][0]:.2f}%')
print(f'Log percent: {sin2_2th_log_p_current:.3g}%, Linear percent: {100*sin2_2th_log_p[sin2_2th_current_idx][0]:.2f}%')

ax1.set_title('m_d', weight='bold')
ax2.set_title('sin2_2th', weight='bold')
ax1.set_ylabel('percentage %', fontsize=16)
ax2.set_ylabel('percentage %', fontsize=16)

ax1.axvline(m_d_grid[md_current_idx], color='k', ls='--')
ax2.axvline(sin2_2th_grid[sin2_2th_current_idx], color='k', ls='--')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.grid(True)
ax2.grid(True)

ax1.legend()
ax2.legend()
fig.tight_layout()
plt.show()