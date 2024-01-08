"""
Probability people share birthday
"""

N = input('Number of people: ')      # Number of people in same room 
N = int(N)
y = 365     # Days in a year


S = 1
for n in range(N):
    S *= (y-n) / y

P = (1 - S)*100     # Probability of at least to of the N persons having same birthday
print(f'Probability: {P}%') 