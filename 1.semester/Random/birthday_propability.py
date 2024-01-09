"""
Probability people share birthday.
Imagine a room with N people. The probability 
that at least two people share birthday is P = 1 - NP, 
where NP is the probability that noone shares birthday. 
The probability that person 1 has a birthday is 365/365. 
Now, the probability of person 2 not having the same birthday 
is  364/365. This goes on to person N, which then have probability
(365 - (N-1)) / 365. Since these events are assumed to not be 
related, the probability of noone sharing birthday is 
NP = 365/365 * 364/365 * 363/365 * ... * (365 - (N-1))/365  
   = 1 / (365)^N * 365! / (365 - N)!
   = 1 / (365)^N * (365 choose N) * N!

And then P = 1 - NP. 
However, this formula is not very convenient for computers, since
they tend to hate factorials. Thus a loop with a product will suffice. 
"""

N = input('Number of people: ')      # Number of people in same room 
N = int(N)
y = 365     # Days in a year


S = 1
for n in range(N):
    S *= (y-n) / y

P = (1 - S)*100     # Probability of at least to of the N persons having same birthday
print(f'Probability: {P}%')