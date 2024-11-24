import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors 

"""
Two ways to do it:
One using contour to draw contour 
Other using imshow to draw contour 
"""

data = np.random.random((10,10))
X = np.linspace(-2, 2, 1000)
Y = np.linspace(-1, 1, 1000)
x, y = np.meshgrid(X, Y, indexing='ij')
f = x**2 + y**2

fig, ax = plt.subplots()
# im = ax.imshow(data, cmap='jet', interpolation='bilinear')
im = plt.contourf(x, y, f, cmap='jet', levels=300)
# im = plt.imshow(f, extent=[X.min(),X.max(),Y.min(),Y.max()], interpolation='bilinear', cmap='jet')
cbar = fig.colorbar(im)
ax.set_title('Click on the colorbar')

highlight_cmap = colors.ListedColormap(['k'])
highlight = ax.imshow(np.ma.masked_all_like(f), interpolation='nearest', vmin=f.min(), vmax=f.max(), extent=[X.min(),X.max(),Y.min(),Y.max()], cmap=highlight_cmap, origin='lower', aspect='auto', zorder=10)

# For contour
highlight = [ax.contour(x, y, f, colors='none')]

def on_pick(event):
    val = event.mouseevent.ydata
    selection = np.ma.masked_outside(f, val-0.03, val + 0.03)
    # highlight.set_data(selection.T)
    highlight[0].remove()
    highlight[0] = ax.contour(x, y, selection, colors='k')
    fig.canvas.draw()

cbar.ax.set_picker(True)
fig.canvas.mpl_connect('pick_event', on_pick)

# ax.imshow(f, origin='lower')

plt.show()