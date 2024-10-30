import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib as mpl

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "euclid"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

N = int(1e5)
x = np.linspace(1e-1, 1e2, N)
xlog = np.log(x)
BE = 1 / (np.exp(x) + 1)

def get_freeze_out(freeze_out=1e-6):

    norm_factor = np.exp(x[0])*(1 / (np.exp(x[0]) + 1) - freeze_out)

    chem_dec = np.log(np.log(1/2*((1-norm_factor)/freeze_out - 1 + np.sqrt((1 + (norm_factor-1)/freeze_out)**2 - 4*norm_factor/freeze_out))))

    FO = (norm_factor*np.exp(-x) + freeze_out)*(xlog > chem_dec) + BE*(xlog < chem_dec)

    return FO

def get_freeze_in(freeze_in=1e-4):

    m = 1.5       # some sort of steepness parameter 
    p = 0.5
    a = m*np.log(freeze_in*1e8) / (1 + 2*m + p*m)
    b = np.exp(2*a)*1e-8
    d = np.exp(m*p)*(np.log(freeze_in) - a*p - np.log(b))

    FI = b*x**(a)*(xlog<=p)
    FI = FI + freeze_in*np.exp(-d*x**(-m))*(xlog>p)

    return FI

def get_pandemic_dark_matter(freeze_PDM=1e-5):

    a = 8
    q = 1
    d = ( (np.log(1e-8))*(1 + np.exp(a*(2+q))) - np.log(freeze_PDM) )*np.exp(-a*(2+q))
    b = 0.8
    PDM = np.exp((np.log(freeze_PDM) - d - b*(xlog+2)[xlog<q][-1]) / (1 + np.exp(-a*(xlog-q))) + d + b*(xlog+2)*(xlog<q) + b*(xlog+2)[xlog<q][-1]*(xlog>q))
    
    return PDM



fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()

# ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(xlog, BE, 'r')
# Freeze-out 
fo_c = 'k'
ax.plot(xlog, get_freeze_out(1e-5), color=fo_c, label=r'Freeze-out', linestyle='--', alpha=0.8)
ax.plot(xlog, get_freeze_out(1e-6), color=fo_c, linestyle='--', alpha=0.8)
# Freeze-in
fi_c = 'k'
ax.plot(xlog, get_freeze_in(5e-6), color=fi_c, label=r'Freeze-in', linestyle=':')
ax.plot(xlog, get_freeze_in(5e-7), color=fi_c, linestyle=':')
# Pandemc 
pd_c = 'k'
ax.plot(xlog, get_pandemic_dark_matter(2.5e-5), color=pd_c, label=r'Pandemic', linestyle='-.')

ax.set_ylim(1e-8, 9)
ax.set_xlim(-3)

ax.text(-1.8, 3.5, r'$Y_{\chi}\propto a^3n_{\chi}$', fontsize=16)
ax.text(3.8, 2e-8, r'$x=m_{\chi}/T$', fontsize=16)

ax.arrow(x=3.5, y=1e-5, dx=0, dy=-(1e-5-1e-6), color='b', width=2.3e-2, head_width=1.2e-1, head_length=8e-7, length_includes_head=True)
ax.text(3.7, 1.7e-6, r'$\boldsymbol{\langle\sigma v\rangle}$', color='b', fontsize=16)

ax.arrow(x=1.5, y=4.7e-7, dx=0, dy=(3.77e-6-4.7e-7), color='b', width=2.3e-2, head_width=1.2e-1, head_length=1.5e-6, length_includes_head=True)
ax.text(1.7, 9e-7, r'$\boldsymbol{\langle\sigma v\rangle}$', color='b', fontsize=16)

ax.text(1.7, 1e-2, r'$n_{\chi, \rm eq}$', color='r', fontsize=18)

ax.set_xticks([1], labels=[r'$\boldsymbol{1}$'], fontsize=16)
ax.set_yticklabels([])
ax.spines['left'].set_position(('data', -2))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.plot(1, 1e-8, ">k", transform=ax.get_yaxis_transform(), ms=5, clip_on=False)
ax.plot(-2, 1, "^k", transform=ax.get_xaxis_transform(), ms=5, clip_on=False)

ax.legend(prop={'size': 14}, frameon=False)
fig.tight_layout()
plt.savefig('freeze_in_out_PDM.pdf')
plt.show()