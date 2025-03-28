import numpy as np
import matplotlib.pyplot as plt 

# plt.axis([1, 100, 0, 1])

yold = None
for i in range(1, 100):
    ynew = np.random.random()
    plt.semilogx([i, i+1], [yold, ynew], linestyle='-')
    plt.pause(0.05)
    yold = ynew
print('Done!')

# without show, the plot will not stay after loop
plt.show()