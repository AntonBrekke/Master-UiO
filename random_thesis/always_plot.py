import numpy as np
import matplotlib.pyplot as plt 

run_loop = True
while run_loop:
    load_str = input('')
    if load_str=='' or load_str[0]!='A':
        exit()

    # load_str = 'md_1.08264e-05;mX_3.24791e-05;sin22th_1.1721e-16;y_3.8676e-03;full_new.dat'
    # load_str = 'A_1.2;B_3.2;C_15;D_3.8;full_new.dat'
    var_list = load_str.split(';')[:-1]
    md, mX, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
    A = md
    B = mX
    C = sin22th
    D = y

    x = np.linspace(0, 1, 1000)
    func = A + np.exp(-B*x)*np.sin(C*x)*D

    plt.plot(x, func)
    plt.show()

