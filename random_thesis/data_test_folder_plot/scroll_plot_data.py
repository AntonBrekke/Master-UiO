from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 

directory_in_str = './'

plots_d = []
plots_X = []
var_list_keep = []

pathlist = Path(directory_in_str).glob('**/*.dat')
for path in pathlist:
    # because path is object not string
    load_str = str(path)
    # load_str_list.append(path_in_str)

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

    plots_d.append((x1_tr, y1_tr))
    plots_X.append((x1_tr, mX*nX/ent))
    var_list_keep.append((md, mX, sin22th, y))

# print(load_str_list)

curr_pos = 0

def key_event(e):
    global curr_pos

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots_d)

    ax.cla()
    ax.loglog(plots_d[curr_pos][0], plots_d[curr_pos][1],  color='#7bc043')
    ax.loglog(plots_X[curr_pos][0], plots_X[curr_pos][1],  color='#f37736')

    md, mX, sin22th, y = var_list_keep[curr_pos]
    ax.set_title(f'md: {md:.1e}, mX: {mX:.1e}, sin22th: {sin22th:1e}, y: {y:.1e}')

    ax.set_xlim(2e-5, 20)
    ax.set_ylim(1e-25, 2e-8)
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
md, mX, sin22th, y = var_list_keep[0]
ax.set_title(f'md: {md:.1e}, mX: {mX:.1e}, sin22th: {sin22th:.1e}, y: {y:.1e}')
ax.loglog(plots_d[0][0], plots_d[0][1],  color='#7bc043')
ax.loglog(plots_X[0][0], plots_X[0][1],  color='#f37736')
ax.set_xlim(2e-5, 20)
ax.set_ylim(1e-23, 2e-10)
plt.show()
