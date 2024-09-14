import numpy as np
import matplotlib.pyplot as plt

"""
A=5, B=4, C=3, D=2, E=1, F=0
"""
scores = np.array([0, 4, 0, 1, 3, 2, 0, 4, 4, 0, 4, 2, 5, 3, 3, 3])

plt.style.use('ggplot')

counts, bins = np.histogram(scores, bins=np.arange(0, 7, 1))   
N, bins, patches = plt.hist(bins[:-1], bins, weights=counts, edgecolor='k', align='left', facecolor='tab:green')
print(len(scores), np.sum(counts), np.mean(scores))
plt.xticks(np.arange(0, 7, 1))
plt.title(f'Total: {np.sum(counts)}', fontsize=16)
plt.gca().invert_xaxis()
plt.axvline(np.mean(scores), color='k', linestyle='--', label=f'mean: {np.mean(scores):.3f}')
plt.legend(loc='lower right')
plt.show()