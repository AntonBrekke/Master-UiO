import numpy as np
import scipy.constants as scc

c = scc.c                   # Speed of light
G = scc.G                   # Newton gravitation
R_earth = 6371e3            # Earth radius
R_sat = R_earth + 20e6      # + 20 000 km
M_earth = 5.97219e24        # Earth mass, kg 
dt_earth = 86400            # 1 day in seconds
v = 1e7 / 3600              # 10 000 km/h
gamma = 1 / np.sqrt(1 - (v/c)**2)

def phi(r):
    return -G*M_earth/r

delta_time = (1/gamma + np.sqrt((c**2 + 2*phi(R_sat))/(c**2 + 2*phi(R_earth))) - 2)*dt_earth
print(f'\nTotal difference in time due to GR and SR: {delta_time:.3e}')