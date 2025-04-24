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

# Anton: Formatting is to move y-tick scale from GeV to keV (str(int(i+6)))
ytMajor = np.array([np.log10(10**j) for j in np.linspace(-25, 5, 25-(-5)+1)])
ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
ylMajor = [r"$10^{" + str(int(i+6)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
ytMajor = 10**ytMajor
ytMinor = 10**ytMinor
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

"""
Had to fix: the right entropy 'ent' was not returned, which caused trouble in plots. 
Should be fixes now. 

data = np.loadtxt('./md_1e-5_mX_2.5e-5_sin22th_1e-12_y_5.6e-5_full.dat')
0: t_grid 
1: T_SM_grid
2: T_nu_grid
3: ent_grid
4: hubble_grid
5: sf_grid / sf_grid
6: T_chi_grid_sol
7: xi_chi_grid_sol
8: xi_X_grid_sol
9: n_chi_grid_sol
10: n_X_grid_sol

data: t, T_SM, T_nu, ent, H, sf, T_d, xi_d, xi_X, xi_h, n_d, n_X, n_h
"""

load_str = './md_9.41205e-05;mX_2.82361e-04;sin22th_6.23551e-17;y_1.11318e-02;full.dat'
load_str = './md_9.41205e-05;mX_2.82361e-04;sin22th_6.61474e-16;y_3.50346e-03;full.dat'
load_str = './md_5.13483e-05;mX_1.54045e-04;sin22th_1.19378e-15;y_2.23145e-03;full_new.dat'
load_str = './md_2.15e-05;mX_6.45e-05;sin22th_1.2e-15;y_1.9e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;sin22th_3e-15;y_1.5e-03;full_new.dat'
load_str = './md_2.1503e-05;mX_6.4509e-05;sin22th_1.32739e-15;y_1.77827e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_3e-04;sin22th_1.32739e-15;y_1.77827e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_8e-05;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_5e-05;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-07;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-10;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-02;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_1e-02;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_2e-08;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_4e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_5e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_4e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1.3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.4e-04;sin22th_1e-15;y_1e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.4e-04;sin22th_1e-15;y_2e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.4e-04;sin22th_1e-15;y_3e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_4e-04;sin22th_1e-15;y_3e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.2e-03;sin22th_1e-15;y_3e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.8e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1.9e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1.3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_1.3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_1e-07;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_1e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_3e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_3e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_2e-03;sin22th_1e-15;y_1e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-03;sin22th_1e-15;y_1e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.18e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.202e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.22e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_5e-05;mh_1.02e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_2.02e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_4.02e-05;mh_8.0802e-05;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_3e-03;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_1.99e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1.7e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.206e-04;sin22th_1e-15;y_1e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.206e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.206e-04;sin22th_1e-15;y_2e-06;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-13;y_2e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-13;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-13;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2.3e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2.5e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_1.2e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_1.7e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2.5e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1.1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-15;y_1.15e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-16;y_1.3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-16;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-16;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_1e-03;mh_6e-05;sin22th_1e-15;y_2.3e-03;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_6e-05;sin22th_1e-15;y_1.8e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_6e-05;sin22th_1e-15;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_6e-05;sin22th_1e-15;y_1.3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.194e-04;sin22th_1e-13;y_2.3e-04;full_new.dat'      # **
load_str = './md_2e-05;mX_1.4e-04;mh_2.786e-04;sin22th_1e-15;y_1.5e-03;full_new.dat'        # interesting 
load_str = './md_2e-05;mX_1.4e-04;mh_1.68e-04;sin22th_1e-15;y_1.5e-03;full_new.dat'         # interesting 
load_str = './md_2e-05;mX_1.4e-04;mh_6e-05;sin22th_1e-15;y_1.8e-03;full_new.dat'         # interesting 
load_str = './md_2e-05;mX_2e-04;mh_1e-04;sin22th_1e-15;y_1.68e-03;full_new.dat'     # Nice
load_str = './md_2e-05;mX_2e-04;mh_4.02e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_1e-04;sin22th_1e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_1e-04;sin22th_1e-15;y_1.6e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_1e-04;sin22th_1e-15;y_1.7e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_1e-04;sin22th_1e-15;y_1.65e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_1e-04;sin22th_1e-15;y_1.67e-03;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_2e-04;sin22th_1e-15;y_1.68e-03;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_2e-04;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_6e-05;sin22th_1e-15;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_6e-05;sin22th_1e-15;y_1.8e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-04;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_1.2e-03;mh_6e-05;sin22th_1e-15;y_2.7e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-04;mh_6e-05;sin22th_1e-15;y_2.2e-03;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_6e-05;sin22th_1e-15;y_1.8e-03;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.206e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.2e-04;sin22th_1e-15;y_1e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.2e-04;sin22th_1e-15;y_5e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.2e-04;sin22th_1e-15;y_6e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.20001e-04;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_8e-03;sin22th_1e-15;y_1e-05;full_new.dat'
load_str = './md_2e-05;mX_4e-04;mh_8e-04;sin22th_1e-15;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_3e-04;sin22th_1e-15;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_3e-04;sin22th_1e-15;y_1.6e-03;full_new.dat'
load_str = './md_2e-05;mX_2e-04;mh_3e-04;sin22th_1e-15;y_1.7e-03;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_2e-05;sin22th_1e-14;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.4e-04;sin22th_1e-14;y_1e-07;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_1.4e-04;sin22th_1e-14;y_5e-07;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-14;y_1e-07;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-14;y_3e-07;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-12;y_1e-07;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-12;y_1e-08;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-12;y_3e-08;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-1;y_1e-09;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-11;y_5e-09;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-11;y_6e-09;full_new.dat'
load_str = './md_2e-05;mX_1e-04;mh_3e-04;sin22th_1e-11;y_8e-09;full_new.dat'
load_str = './md_2e-05;mX_1e-03;mh_6e-05;sin22th_1e-15;y_2.6e-03;full_new.dat'          # interesting 
load_str = './md_2e-05;mX_1e-04;mh_6e-05;sin22th_1e-15;y_1.5e-03;full_new.dat'          # interesting 
load_str = './md_2e-05;mX_6e-05;mh_5e-05;sin22th_1e-16;y_3e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_5e-05;sin22th_5e-16;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_5e-05;sin22th_5e-16;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_5e-05;mh_6e-05;sin22th_5e-16;y_1.5e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_5e-05;sin22th_5e-16;y_1.8e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_3e-15;y_1.6e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_3e-15;y_8e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_3e-15;y_9e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_3e-15;y_9e-04;full_new.dat'
load_str = './md_2e-05;mX_5e-05;mh_1e-04;sin22th_3e-15;y_2e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_2e-15;y_1.1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_2e-15;y_1e-03;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_2e-15;y_1.05e-03;full_new.dat'
load_str = './md_1.35388e-06;mX_6.76938e-06;mh_4.06163e-06;sin22th_7.01704e-15;y_4.45923e-04;full_new.dat'
load_str = './md_2e-05;mX_6e-05;mh_6e-05;sin22th_3e-15;y_8.98e-04;full_new.dat'     # Perfect
load_str = './md_2e-05;mX_6e-05;mh_5e-05;sin22th_3e-15;y_9.12e-04;full_new.dat'         # Perfect
load_str = './md_2e-05;mX_1e-04;mh_6e-05;sin22th_1e-15;y_1.51e-03;full_new.dat'     # Perfect

data = np.loadtxt(load_str)

T_SM = data[:,1]
T_nu = data[:,2]
ent = data[:,3]
Td = data[:,6]
xid = data[:,7]
xiX = data[:,8]
xih = data[:,9]
nd = data[:,10]
nX = data[:,11]
nh = data[:,12]

c1 = '#7bc043'      # green
c2 = '#f37736'      # orange
c3 = '#13b9af'      # blue

# Mass: 1e-6 * X GeV = X keV
var_list = load_str.split(';')[:-1]
md, mX, mh, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
print(f'md: {md:.2e}, mX: {mX:.2e}, mh: {mh:.2e}, sin22th: {sin22th:.2e}, y: {y:.2e}')
mY_relic = cf.omega_d0 * cf.rho_crit0_h2 / cf.s0        # m*Y = m*n/s = Omega * rho_c0 / s0

T_grid_dw = np.logspace(np.log10(1.4e-3), 1, 400)
mYd_dw = cf.O_h2_dw_Tevo(T_grid_dw, md, 0.5*np.arcsin(np.sqrt(sin22th)))*cf.rho_crit0_h2 / cf.s0     # Anton: mY from Dodelson-Widrow

if True:

    x1_dw = md/T_grid_dw
    y1_dw = mYd_dw

    x1_tr = md/T_nu
    y1_tr = md*nd/ent

    x1_dw0, x1_tr0 = x1_dw[x1_dw < 1e-3], x1_tr[x1_tr > 1e-3]
    y1_dw0, y1_tr0 = y1_dw[x1_dw < 1e-3], y1_tr[x1_tr > 1e-3]

    x1, y1 = np.array([*x1_dw0[::-1], *x1_tr0, 1e3]), np.array([*y1_dw0[::-1], *y1_tr0, y1_tr0[-1]])

    ax1.loglog(x1, y1, color=c1, zorder=-1)
    # ax1.loglog(x1_tr, y1_tr, color='#7bc043', zorder=-1)
    # ax1.loglog(x1_dw[x1_dw < 1e-3], y1_dw[x1_dw < 1e-3], color='r', zorder=-1)

    ax1.fill_betweenx([1e-28, 1e5], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
    # ax1.fill_betweenx([1e-23, 1e-18], 1e-5, 1e-3, facecolor="white", hatch="\\", edgecolor="0.9", zorder=1)

    #ax1.text(8e-5, 2e-21, 'Thermalization', fontsize=10, color='darkorange')
    #ax1.text(5e-4, 1.3e-19, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    #ax1.text(5e-4, 1e-22, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    ax1.axvline([1e-3], ls=':', color='0', zorder=-2)
    ax2.axvline([1e-3], ls=':', color='0', zorder=-2)
    # ax1.plot([1e-3]*2, [1e-25, 2e-8], ls=':', color='0', zorder=-2)
    # ax2.plot([1e-3]*2, [1e-2, np.max(Td/T_nu)*1.5], ls=':', color='0', zorder=-2)

    ax1.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
    #ax1.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


    ax1.loglog(md/T_nu, mh*nh/ent, color=c2, ls='-', zorder=-4)
    ax1.loglog(md/T_nu, mX*nX/ent, color=c3, ls='-', zorder=-4)

    ax1.loglog([1e-8, 1e3], [mY_relic, mY_relic], color='0.55', ls='-.', zorder=-2)
    ax1.text(3e-5, 1e-11, r'$\Omega_s h^2 = 0.12$', fontsize=11, color='0.55')

    YX_max = np.max(mX*nX/ent)
    Yh_max = np.max(mh*nh/ent)
    Ys_max = np.max(y1)
    ax1.text(md/T_nu[np.where(mX*nX/ent==YX_max)], Ys_max*1e-2, r'$\nu_s$', color=c1, fontsize=11)
    ax1.text(md/T_nu[np.where(mX*nX/ent==YX_max)], YX_max*1e-2, r'$X$', color=c3, fontsize=11)
    ax1.text(md/T_nu[np.where(mh*nh/ent==Yh_max)]*3, Yh_max*1e-1, r'$h$', color=c2, fontsize=11)

    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-' , color='black', label=r'$\text{BP1}$')
    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\text{BP2}$')

    ax2.loglog(md/T_nu, Td/T_nu, color='0.4', ls='-', zorder=-4)

    ax2.fill_betweenx([1e-1, 1.5e0], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")

    #plt.text(2e-2, 1e-11, r'$m_X = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props)
    #plt.text(1e0, 3e-23, r'$m_X = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props, zorder=5)
    #ax1.text(1e0, 8e-25, r'$m_X = 2.5m_\chi$', fontsize=10, horizontalalignment='center', zorder=5)


    ax2.legend(fontsize=10, framealpha=0.8, edgecolor='1')
    ax2.xaxis.set_label_text(r"$m_s / T_\nu$")
    ax1.yaxis.set_label_text(r"$m\, n / s\;\;\mathrm{[keV]}$")
    ax2.yaxis.set_label_text(r"$T_\text{d}/T_\nu$")


    ax1.xaxis.set_major_locator(xMajorLocator)
    ax1.xaxis.set_minor_locator(xMinorLocator)
    ax1.xaxis.set_major_formatter(xMajorFormatter)
    ax1.yaxis.set_major_locator(yMajorLocator)
    ax1.yaxis.set_minor_locator(yMinorLocator)
    ax1.yaxis.set_major_formatter(yMajorFormatter)

    plt.xlim(2e-5, 20)

    # ylim + 6 will be shown
    ax1.set_ylim(1e-25, 2e-8)
    ax2.set_ylim(np.min(Td/T_nu)*0.5, np.max(Td/T_nu)*1.5)
    # ax1.set_title(fr'$\sin^2(2\theta)$={sin22th:.1e}, $y$={y:.1e}\\$m_d$={md:.1e}, $m_X$={mX:.1e}, $m_h$={mh:.1e}', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    # plt.savefig(f'dens_evo_{load_str.replace("./", "").replace(".dat","")}_final.pdf')
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
