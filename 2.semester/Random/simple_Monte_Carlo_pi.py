import numpy as np

r = 1
N = 1e8
X = np.random.uniform(-r, r, [int(N), 2])
C = X[X[:,0]**2 + X[:,1]**2 <= r**2]

print(f'Approximate pi: {4*C.size/X.size}')