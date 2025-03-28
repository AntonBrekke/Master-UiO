import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def fill_nan(data, method='linear'):
    # Create x and y coordinate grids for the data
    ny, nx = data.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Flatten the arrays for use with griddata
    x_flat = x.flatten()
    y_flat = y.flatten()
    data_flat = data.flatten()
    
    # Identify valid (non-NaN) points
    valid = ~np.isnan(data_flat)
    points_valid = np.vstack((x_flat[valid], y_flat[valid])).T
    values_valid = data_flat[valid]
    
    # Points where data is NaN
    points_missing = np.vstack((x_flat[~valid], y_flat[~valid])).T
    
    # Interpolate the missing data
    data_flat[~valid] = griddata(points_valid, values_valid, points_missing, method=method)
    
    # Reshape back to the original data shape
    return data_flat.reshape(data.shape)

# Example: Create a sample 2D array with some NaNs
data = np.array([[1, 2, 3, 4],
                 [2, np.nan, np.nan, 5],
                 [3, 4, 5, 6],
                 [4, 5, 6, 7]], dtype=float)

X = np.linspace(-1, 1, 50)
Y = np.linspace(-1, 1, 50)

x, y = np.meshgrid(X, Y)
z = x**2 + y**2
z[x**2 + y**2 < 0.5] = np.nan

data = z

# Fill the NaNs using linear interpolation
filled_data = fill_nan(data, method='nearest')

print("Original Data:\n", data)
print("Filled Data:\n", filled_data)

# Optional: visualize the original and filled data
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].contourf(x, y, data, levels=100)
axs[1].contourf(x, y, filled_data, levels=100)
plt.show()
