from math import factorial as fac
import numpy as np

"""
Give number of integer solutions of equation 
a*x_1 + a*x_2 + ... + a*x_r1 + b*x_(r1+1) + ... + b*x_(r1+r2=m) = n

The lists 
C = [a, b] or C = [b, a]
r = [r1, r2] or r = [r2, r1], correspondingly.

Can e.g. get number of possible terms in cross-sections: 
Have variables s, t, u, m1, m2, and final result could have e.g. mass-dim 8. 
Then all combinations of terms that could show up should satisfy 
2*[s] + 2*[t] + 2*[u] + [m1] + [m2] = 8
with n=8, a=2, r1=3, b=1, r2=2 

integer_solutions implements 
sum_(k_1=0)^(inf)*sum_(k_2=0)^(inf) B(r1+k1-1, k1) * B(r2+k2-1, k2) * (a2*k2 == n - a1*k1)
which should be symmetric in k1 and k2, (a2*k2 == n - a1*k1) = (a1*k1 == n - a2*k2)
"""

def B(n, k):
    return fac(n)/(fac(k)*fac(n-k))

def integer_solutions(n, C, r):
    s = 0 
    for k in range(0, int(np.floor(n/C[0]))+1):
        # Only integer values for k2, else delta function yields 0 
        if not (1/C[1]*(n-C[0]*k)).is_integer():
            s += 0 
            continue
        s += B(r[0] + k - 1, k)*B(r[1] + int(1/C[1]*(n-C[0]*k))-1, int(1/C[1]*(n-C[0]*k)))
        
    return s

print(integer_solutions(n=10, C=[2, 1], r=[3, 6]))