# from tkinter import *
import tkinter as tkint
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 

lst = [f'path {i}' for i in range(1,6)]

directory_in_str = './'
directory_in_str = './Master-UiO/project-code/code/sterile_test'
directory_in_str = 'C:/Users/anton/Desktop/Python/Master-UiO/project-code/code/sterile_test'

load_str_list = []

pathlist = Path(directory_in_str).glob('**/*.dat')
for path in pathlist:
    # because path is object not string
    load_str = str(path)
    load_str_list.append(load_str)


root = tkint.Tk()
def select(e):
    load_str = e.widget.get(*e.widget.curselection())
    load_str = directory_in_str + '/' + load_str
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot()

    data = np.loadtxt(load_str)

    T_SM = data[:,1]
    T_nu = data[:,2]
    ent = data[:,3]
    Td = data[:,6]
    xid = data[:,7]
    xiX = data[:,8]
    nd = data[:,9]
    nX = data[:,10]

    var_list = load_str.split(';')[:-1]
    md, mX, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
    # print(f'md: {md:.2e}, mX: {mX:.2e}, sin22th: {sin22th:.2e}, y: {y:.2e}')

    x1_tr = md/T_nu
    y1_tr = md*nd/ent

    ax.loglog(x1_tr, y1_tr,  color='#7bc043')
    ax.loglog(x1_tr, mX*nX/ent,  color='#f37736')
    ax.set_title(f'md: {md:.1e}, mX: {mX:.1e}, sin22th: {sin22th:1e}, y: {y:.1e}')

    ax.set_xlim(2e-5, 20)
    ax.set_ylim(1e-25, 2e-8)
    fig.tight_layout()
    plt.show()


lstbox = tkint.Listbox(root, width=75, height=40)
lstbox.pack(padx=10, pady=10, fill='both', expand=True)

for i in load_str_list:
    # redefinition of i not needed if directory_in_str = './'
    lstbox.insert('end', i.split('\\')[-1])

lstbox.bind('<<ListboxSelect>>',select) # Or lstbox.bind('<Double-1>',select) for double click

root.mainloop()