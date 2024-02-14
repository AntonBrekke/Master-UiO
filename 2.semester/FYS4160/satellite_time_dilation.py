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

def delta_time_tot(r_earth, r_sat, dt_earth=86400):
    delta = (1/gamma + np.sqrt((c**2 + 2*phi(R_sat))/(c**2 + 2*phi(R_earth))) - 2)*dt_earth
    return delta

delta_time = delta_time_tot(R_earth, R_sat, dt_earth)
print(f'\nTotal difference in time due to GR and SR: {delta_time:.3e} s')
print(f'Length scale given light speed signals: {delta_time*c:.3e} m')

# Say I want 15m precicion on GPS. How long will it take before it messes up?
meter_precision = 15            # m
dt_earth = meter_precision/(c*(1/gamma + np.sqrt((c**2 + 2*phi(R_sat))/(c**2 + 2*phi(R_earth))) - 2))
print(f'Given 15m precicion on GPS, without relativic correction: {dt_earth:.2f} s')