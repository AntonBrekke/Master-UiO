import numpy as np
import matplotlib.pyplot as plt 

N = int(1e4)
x = np.linspace(-8, 8, N)

def f(x):
    return x**2 - 4

def df(x):
    return 2*x

f = f(x)
df = df(x)
print(df)
df = np.gradient(f, x[1]-x[0])
print(df)

def Newton_method(x, f, df, xi, eps=1e-2):
    dx = x[1] - x[0]
    x0 = x[0]
    index = int((xi - x0) / dx)
    xn = xi
    while abs(f[index]) > eps:
        xn = x[index] - f[index] / df[index]
        index = int((xn - x0) / dx)
        print(f[index])
    return xn

print(Newton_method(x, f, df, 5))