import numpy as np
import matplotlib.pyplot as plt 

# Function that finds indices of data satisfying a given condition
def find_index(data, condition, start=None, end=None):
    if start is None: start = np.min(data)
    if end is None: end = np.max(data)
    domain = (data > start)*(data < end)
    condition = condition[domain]

    range_array = np.array(range(len(condition)))
    index_satisfied_condition = range_array[condition]
    data_satisfied_condition = data[domain][index_satisfied_condition]

    min_val = np.min(data_satisfied_condition)
    max_val = np.max(data_satisfied_condition)

    if abs(min_val - max_val) < 1e-2:
        return np.mean([min_val, max_val])
    else:
        return np.array([min_val, max_val])


# Function searching for index given condition G < 0
def find_index_v2(data, condition, start=None, end=None, eps=0.1, steps=1e-5):
    if start is None: start = np.min(data)
    if end is None: end = np.max(data)
    domain = (data > start)*(data < end)
    condition = condition[domain] < eps

    range_array = np.array(range(len(condition)))
    index_satisfied_condition = range_array[condition]
    data_satisfied_condition = data[domain][index_satisfied_condition]

    min_val = np.min(data_satisfied_condition)
    max_val = np.max(data_satisfied_condition)

    if abs(min_val - max_val) < 1e-2:
        return np.mean([min_val, max_val])
    else:
        return np.array([min_val, max_val])
    
# Less scuffed version of last index-finders
def find_index_v3(x, data, value, x_start=None, x_end=None):
    if x_start is None: x_start = np.min(x)
    if x_end is None: x_end = np.max(x)
    domain = (x >= x_start) * (x <= x_end)
    data_domain = data[domain]
    index = np.where(np.min(abs(data_domain - value)) == abs(data_domain - value))[0][0]
    index = len(x[(x <= x_start)]) + index      # Translate to index on original domain
    return index
# End of index-function

N = 10001
x = np.linspace(-10, 10, N)
f = 0.1*x**3 + 5*x**2 + 4*x - 4

plt.plot(x, f)
plt.show()

# Find roots of f: 
index = find_index_v3(x, f, value=0)    
print(x[index], f[index])