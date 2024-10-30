import numpy as np

# Weinberg angles
sw = np.sqrt(0.223)
cw = np.sqrt(0.777)

# Gauge boson masses GeV/c^2. 1MeV = 1e-3GeV, 1eV = 1e-9GeV
mZ = 91.1876    
mX = 200        # Heavier than any SM particle to test boundaries

# Mixing parameter (0 < sigma < 1, typically ~ 10^(-12) - 10^(-3))
sigma = 1e-3

"""
tan(2*theta) = 2*M12 / (M11 - M22) 
in mass matrix between Z and X boson
"""
M11 = (mZ**2*(1 - (sigma*cw)**2)**2 + (mX*sigma*sw)**2) / ((1-sigma**2)*(1-(sigma*cw)**2))
M12 = mX**2*sigma*sw / (np.sqrt(1-sigma**2)*(1-(sigma*cw)**2)) 
M22 = mX**2 / (1 - sigma**2*cw**2)
theta = 1/2*np.arctan2(2*M12, M11 - M22)

print(f'theta: {theta * 180 / np.pi:.3e}deg')

X_mix_EM = sigma*cw / (np.sqrt(1-(sigma*cw)**2))*np.cos(theta) - sigma**2*sw*cw / (np.sqrt(1-sigma**2)*np.sqrt(1-(sigma*cw)**2))*np.sin(theta)
Z_mix_EM = sigma*cw / (np.sqrt(1-(sigma*cw)**2))*np.sin(theta) + sigma**2*sw*cw / (np.sqrt(1-sigma**2)*np.sqrt(1-(sigma*cw)**2))*np.cos(theta)

Z_mix_Z = np.sqrt((1-(sigma*cw)**2) / (1 - sigma**2))*np.cos(theta)
X_mix_Z = np.sqrt((1-(sigma*cw)**2) / (1 - sigma**2))*np.sin(theta)

Z_mix_X = sigma*sw / (np.sqrt(1-sigma**2)*np.sqrt(1-(sigma*cw)**2))*np.cos(theta) + 1 / np.sqrt(1-(sigma*cw)**2)*np.sin(theta)
X_mix_X = -sigma*sw / (np.sqrt(1-sigma**2)*np.sqrt(1-(sigma*cw)**2))*np.sin(theta) + 1 / np.sqrt(1-(sigma*cw)**2)*np.cos(theta)

print(f'X mix EM: {X_mix_EM:.4e}')
print(f'Z mix EM: {Z_mix_EM:.4e}')
print(f'X mix Z: {X_mix_Z:.4e}')    # This also shows up in interaction with Higgs 
print(f'Z mix Z: {Z_mix_Z:.4e}')
print(f'Z mix X: {Z_mix_X:.4e}')
print(f'X mix X: {X_mix_X:.4e}')