#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 8.0

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import constants_functions as cf

md1 = 1.2e-5
md2 = 2.0e-5
mphi1 = 3*md1
mphi2 = 3*md2
mY_relic = cf.rho_d0/cf.s0

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white", gridspec_kw={'height_ratios': [2, 1]})
ax1.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax1.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(0.5)
ax2.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax2.spines[axis].set_linewidth(0.5)

xtMajor = np.array([np.log10(10**j) for j in np.linspace(-5, 2, 8)])
xtMinor = np.array([np.log10(i*10**j) for j in xtMajor for i in range(10)[1:10]])
xlMajor = [r"$10^{" + str(int(i)) + "}$" if i in xtMajor else "" for i in xtMajor]
xtMajor = 10**xtMajor
xtMinor = 10**xtMinor
xMajorLocator = FixedLocator(xtMajor)
xMinorLocator = FixedLocator(xtMinor)
xMajorFormatter = FixedFormatter(xlMajor)

ytMajor = np.array([np.log10(10**j) for j in np.linspace(-25, -9, 17)])
ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
ylMajor = [r"$10^{" + str(int(i+6)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
ytMajor = 10**ytMajor
ytMinor = 10**ytMinor
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

"""
data1 = np.loadtxt('./md_1.0000e-05_mphi_2.5000e-05_sin22th_3.5000e-13_y_1.3099e-04.dat')
# data2 = np.loadtxt('./md_2.00e-05_mphi_5.00e-05_sin22th_3.50e-15_y_1.47e-03.dat')
data2 = np.loadtxt('./md_2.0000e-05_mphi_5.0000e-05_sin22th_3.5000e-15_y_1.4740e-03.dat')
"""
data1 = np.loadtxt('./md_1.2000e-05_mphi_3.6000e-05_sin22th_2.5000e-13_y_1.9050e-04.dat')
data2 = np.loadtxt('./md_2.0000e-05_mphi_6.0000e-05_sin22th_3.0000e-15_y_1.6022e-03.dat')

T_SM1 = data1[:,1]
T_SM2 = data2[:,1]
T_nu1 = data1[:,2]
T_nu2 = data2[:,2]
ent1 = data1[:,3]
ent2 = data2[:,3]
Td1 = data1[:,6]
xid1 = data1[:,7]
xiphi1 = data1[:,8]
nd1 = data1[:,9]
nphi1 = data1[:,10]
Td2 = data2[:,6]
xid2 = data2[:,7]
xiphi2 = data2[:,8]
nd2 = data2[:,9]
nphi2 = data2[:,10]

T_grid_dw = np.logspace(np.log10(1.4e-3), 1, 400)
mYd1_dw = cf.O_h2_dw_Tevo(T_grid_dw, md1, 0.5*np.arcsin(np.sqrt(2.5e-13)))*cf.rho_crit0_h2/cf.s0
mYd2_dw = cf.O_h2_dw_Tevo(T_grid_dw, md2, 0.5*np.arcsin(np.sqrt(3.0e-15)))*cf.rho_crit0_h2/cf.s0

if True:

    x1_dw, x2_dw = md1/T_grid_dw, md2/T_grid_dw
    y1_dw, y2_dw = mYd1_dw, mYd2_dw

    x1_tr, x2_tr = md1/T_nu1, md2/T_nu2
    y1_tr, y2_tr = md1*nd1/ent1, md2*nd2/ent2

    x1_dw0, x1_tr0 = x1_dw[x1_dw < 1e-3], x1_tr[x1_tr > 1e-3]
    y1_dw0, y1_tr0 = y1_dw[x1_dw < 1e-3], y1_tr[x1_tr > 1e-3]

    cut = 1e-3
    x2_dw0, x2_tr0 = x2_dw[x2_dw < cut], x2_tr[x2_tr > cut]
    y2_dw0, y2_tr0 = y2_dw[x2_dw < cut], y2_tr[x2_tr > cut]

    x1, y1 = np.array([*x1_dw0[::-1], *x1_tr0, 1e3]), np.array([*y1_dw0[::-1], *y1_tr0, y1_tr0[-1]])
    x2, y2 = np.array([*x2_dw0[::-1], *x2_tr0, 1e3]), np.array([*y2_dw0[::-1], *y2_tr0, y1_tr0[-1]])

    ax1.loglog(x1, y1, color='#7bc043', zorder=-1)
    ax1.loglog(x2, y2, color='#7bc043', linestyle='--', zorder=-1)

    ax1.fill_betweenx([1e-23, 1e-18], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
    #ax1.fill_betweenx([1e-23, 1e-18], 1e-5, 1e-3, facecolor="white", hatch="\\", edgecolor="0.9", zorder=1)

    #ax1.text(8e-5, 2e-21, 'Thermalization', fontsize=10, color='darkorange')
    #ax1.text(5e-4, 1.3e-19, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    #ax1.text(5e-4, 1e-22, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    ax1.plot([1e-3]*2, [1e-25, 1e-9], ls=':', color='0', zorder=-2)
    ax2.plot([1e-3]*2, [1e-2, 2e0], ls=':', color='0', zorder=-2)

    ax1.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
    #ax1.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


    ax1.loglog(md1/T_nu1, mphi1*nphi1/ent1, color='#f37736', ls='-', zorder=-4)
    ax1.loglog(md2/T_nu2, mphi2*nphi2/ent2, color='#f37736', ls='--', zorder=-4)

    ax1.loglog([1e-8, 1e3], [mY_relic, mY_relic], color='0.65', ls='-.', zorder=-2)
    ax1.text(3e-5, 1e-11, r'$\Omega_s h^2 = 0.12$', fontsize=11, color='0.65')

    ax1.text(2.5, 1e-11, r'$\nu_s$', color='#7bc043', fontsize=11)
    ax1.text(2.0, 1e-16, r'$\phi$', color='#f37736', fontsize=11)

    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-' , color='black', label=r'$\textit{BP1}$')
    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\textit{BP2}$')

    ax2.loglog(md1/T_nu1, Td1/T_nu1, color='0.4', ls='-', zorder=-4)
    ax2.loglog(md2/T_nu2, Td2/T_nu2, color='0.4', ls='--', zorder=-4)

    ax2.fill_betweenx([1e-1, 1.5e0], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")

    #plt.text(2e-2, 1e-11, r'$m_\phi = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props)
    #plt.text(1e0, 3e-23, r'$m_\phi = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props, zorder=5)
    #ax1.text(1e0, 8e-25, r'$m_\phi = 2.5m_\chi$', fontsize=10, horizontalalignment='center', zorder=5)


    ax2.legend(fontsize=10, framealpha=0.8, edgecolor='1')
    ax2.xaxis.set_label_text(r"$m_s / T_\nu$")
    ax1.yaxis.set_label_text(r"$m\, n / s\;\;\mathrm{[keV]}$")
    ax2.yaxis.set_label_text(r"$T_\text{d}/T_\nu$")


    #ax1.xaxis.set_major_locator(xMajorLocator)
    #ax1.xaxis.set_minor_locator(xMinorLocator)
    #ax1.xaxis.set_major_formatter(xMajorFormatter)
    ax1.yaxis.set_major_locator(yMajorLocator)
    ax1.yaxis.set_minor_locator(yMinorLocator)
    ax1.yaxis.set_major_formatter(yMajorFormatter)

    plt.xlim(2e-5, 20)

    ax1.set_ylim(1e-25, 1e-9)
    ax2.set_ylim(1e-2, 2e0)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig('evo.pdf')
    plt.show()
elif False:
    plt.loglog(md1/T_nu1, 1e6*Td1, color='dodgerblue', ls='-')
    plt.loglog(md2/T_nu2, 1e6*Td2, color='dodgerblue', ls='--')
    ax.xaxis.set_label_text(r"$m_s / T_\nu$")
    ax.yaxis.set_label_text(r"$T_s \; \; [\mathrm{keV}]$")
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-1, 1e2)
    plt.tight_layout()
    # plt.savefig('Td.pdf')
    plt.show()
else:
    plt.semilogx(md1/T_nu1, xid1, color='dodgerblue', ls='-')
    plt.semilogx(md2/T_nu2, xid2, color='dodgerblue', ls='--')
    ax.xaxis.set_label_text(r"$m_\mathrm{d} / T_\nu$")
    ax.yaxis.set_label_text(r"$\mu_\mathrm{d} / m_\mathrm{d}$")
    ax.set_xlim(1e-3, 1e1)
    # ax.set_ylim(1e-1, 1e2)
    plt.tight_layout()
    # plt.savefig('mud.pdf')
    plt.show()
