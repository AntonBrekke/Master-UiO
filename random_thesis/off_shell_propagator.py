import numpy as np
import matplotlib.pyplot as plt 

def get_figsize(columnwidth, wf=1.0, hf=(5.**0.5-1.0)/2.0):
    """Parameters:
    - wf [float]:  width fraction in columnwidth units
    - hf [float]:  height fraction in columnwidth units.
                    Set by default to golden ratio.
    - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                            using \showthe\columnwidth
    Returns:  [fig_width, fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
params = {'axes.labelsize': 10,
            'axes.titlesize': 10,
            'font.size': 10 } # extend as needed
# print(plt.rcParams.keys())
plt.rcParams.update(params)

"""
In real intermediate state subtraction, we subtract
the on-shell contribution to the Breit-Wigner propagator
|D_BW|^2 --> pi/(M*Gamma) * delta(s - M**2)
by decomposing 
|D_BW|^2 = |D_on|^2 + |D_off|^2
where it is expected that 
|D_on|^2 --> pi/(M*Gamma) * delta(s - M**2)
|D_off|^2 --> 0
in the sense of distributions i.e. when integrated over. 
This is what we check. 
"""

M = 1e-5
Gamma = 1e-18

# Given in the PVS-scheme
def D_BW2(s, M, Gamma):
    propagator_BW = 1/((s-M**2)**2 + (M*Gamma)**2)
    return propagator_BW

def D_RIS2(s, M, Gamma, off_shell=True):
    on_shell = not off_shell
    if off_shell:
        # Off-shell propagator 
        propagator_RIS = ((s-M**2)**2 - (M*Gamma)**2)/((s-M**2)**2 + (M*Gamma)**2)**2
    if on_shell:
        # On-shell propagator
        propagator_RIS = (2*(M*Gamma)**2)/((s-M**2)**2 + (M*Gamma)**2)**2
    return propagator_RIS

# Given in the CUT-scheme. 
# In the cut scheme, prop2 = prop*prop, which is not true in PVS-scheme 
def D_CUT2(s, M, delta, Gamma):
    # e.g. delta = m*Gamma 
    propagator_CUT = (1 - cut_func(s-M**2, delta))**2*D_BW2(s, M, Gamma)
    return propagator_CUT

def cut_func(x, delta):
    # E.g. the top hat function H(delta-x)*H(delta+x)
    return (x < delta)*(x > -delta)

def delta(x, a, eps):
    # pi times delta-function
    repr1 = eps / ((x-a)**2 + eps**2)
    repr2 = 2*eps**3 / ((x-a)**2 + eps**2)**2
    return repr1


# x = np.linspace(-3, 3, int(1e5))
# dx = x[1]-x[0]
# a = 1
# # Integrate over delta should give 1 
# print(1/np.pi*np.sum(delta(x, a, 0.001))*dx)
# plt.plot(x, delta(x, a, 0.1), label='0.1')
# plt.plot(x, delta(x, a, 0.01), label='0.01')
# plt.plot(x, delta(x, a, 0.001), label='0.001')
# plt.legend()
# plt.show()

# Propagator centered around M^2, with width ~ M*Gamma. Choose to plot around 10*width
width_fac = 8
s = np.linspace(M**2-width_fac*M*Gamma, M**2+width_fac*M*Gamma, int(1e4))
ds = s[1]-s[0]
# Integrate BW should give ~1
print(1/np.pi*M*Gamma*np.sum(D_BW2(s, M, Gamma))*ds)
# Integrate off-shell should give ~zero
print(1/np.pi*M*Gamma*np.sum(D_RIS2(s, M, Gamma, off_shell=True))*ds)
# Integrate on-shell should give ~1
print(1/np.pi*M*Gamma*np.sum(D_RIS2(s, M, Gamma, off_shell=False))*ds)

columnwidth = 418.25368     # pt, given by \showthe\textwidth in LaTeX
fig = plt.figure(figsize=get_figsize(columnwidth, wf=1.0), dpi=150, edgecolor="white")
ax1 = fig.add_subplot()
ax1.grid(True)


ggplot_red = "#E24A33"
ch = 'crimson' # crimson
c1 = '#797ef6' # orchid
c2 = '#1aa7ec' # sky blue
c3 = '#4adede' # turquoise
c4 = '#ffa62b' # gold
c5 = '#1e2f97' # dark blue

lw = 2
# Full BW propagator
D_BW_2 = D_BW2(s, M, Gamma)
ax1.plot(s, D_BW_2, color=c5, lw=lw, label=r'$|D_{\rm BW}|^2$')
# Off-shell
D_off_shell = D_RIS2(s, M, Gamma, off_shell=True)
ax1.plot(s, D_off_shell, color=c2, lw=lw, label=r'$|D_{\rm off}|^2$')
# On-shell
D_on_shell = D_RIS2(s, M, Gamma, off_shell=False)
ax1.plot(s, D_on_shell, color=ch, lw=lw, label=r'$|D_{\rm on}|^2$')
# ax2.plot(s, D_CUT2(s, M, M*Gamma, Gamma), color='k', lw=2, label=r'$D_{off}$')

# Width at half maximum
HM1 = 0.5 * np.max(D_BW2(s, M, Gamma))
ax1.plot([M**2 - M*Gamma, M**2 + M*Gamma], [HM1, HM1], color='k', ls='--')

ax1.set_xlim(s[0], s[-1])

# Reduce number of ticks to make it more readable
ax1_xticks = ax1.get_xticks()
import matplotlib 
# ax1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%e"))
ax1.set_xticks(ax1_xticks[::2])

pos_percent_x = 0.8
pos_percent_y_m = 0.10
pos_percent_y_Gamma = 0.03
ymin = np.min(D_off_shell)
ymax = np.max(D_on_shell)
xpos = s[-1]*pos_percent_x+(1-pos_percent_x)*s[0]
ypos_m = ymax * pos_percent_y_m + (1-pos_percent_y_m)*ymin
ypos_Gamma = ymax * pos_percent_y_Gamma + (1-pos_percent_y_Gamma)*ymin
ax1.text(xpos, ypos_m, r'$m=1\,\rm keV$', ha='left')
ax1.text(xpos, ypos_Gamma, r'$\Gamma=10^{-12}\,\rm GeV$', ha='left')
ax1.text(M**2, HM1*0.9, r'$2m\Gamma$', ha='center', va='top')

ax1.set_xlabel(r'$s-m^2\;\;[\mathrm{keV}^2]$')


ax1.legend()
fig.tight_layout()
plt.savefig('RIS_propagators.pdf')
plt.show()