import numpy as np 


a = [1,2,3,4,5,6]

def f(s):
    l = []
    for j in range(s):
        for n in a:
            for m in a:
                if n != m and [n,m] not in l and [m,n] not in l:
                    l.append([n,m])

    return l

print(f(4))