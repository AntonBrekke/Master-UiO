import sympy as sp
s, u, t, a, b = sp.symbols('s u t a b')

# a = 0
# b = 0 
s = 2*a + 2*b - u - t
m = a - b


p1p2, p1q, p1k, p2q, p2k, qk = sp.symbols('p1p2 p1q p1k p2q p2k qk')

""""
p1p2 = 0.5*(s - 2*a)
p1q = 0.5*(t + m)
p1k = 0.5*(u + m)
p2q = -0.5*(t + m)
p2k = -0.5*(u + m)
qk = 2*m
"""

# s = 2*(p1p2 + a)
# t = 2*p1q - m
# u = 2*p1k - m
# m = qk/2
# b = a - m

eq_torsten = ( (s + 2*t - 2*b - 2*a)**2*(-a**2 - b**2 - a*(2*b - s - 2*t) + 2*b*t - t*(s + t)) )

# eq_start = ( (u - a)**2 * (-1/2*(t + m)**2 - t/2*(s - 2*a) - a*(3*t + 2*m + a - 1/2*s) - a**2)
#      + (t - a)**2 * (-1/2*(u + m)**2 - u/2*(s - 2*a) - a*(3*u + 2*m + a - 1/2*s) - a**2)
#      - 2*(t - a)*(u - a) * (-1/2*(t + m)*(u + m) - 1/2*(s - 2*a)*2*m - a*(t + u + 4*m + a - 1/2*s) - a**2)
#    )

eq_start_2 = ( (u - a)**2 * (-1/2*(t + m)**2 - t/2*(s - 2*a) + a*(t - 2*b + 1/2*s))
     + (t - a)**2 * (-1/2*(u + m)**2 - u/2*(s - 2*a) + a*(u - 2*b + 1/2*s))
     + 2*(t - a)*(u - a) * (-1/2*(t + m)*(u + m) - 1/2*(s - 2*a)*2*m + a*(u + t - 2*a + 1/2*s))
   )


eq_me = ( -1/2*(2*a - b)**2*(u-t)**2 - 4*a**2*(u-t)**2 - a*(2*(a-b) - 1/2*s)*(u-t)**2
              -1/2*s*a*(u-t)**2 - 0.5*s*(u + t - 4*(a-b))*(u-a)*(t-a) )

eq_me_2 = ( -1/2*(2*a - b)**2*(u-t)**2 - a*(2*a - 2*b - 1/2*s)*(u-t)**2 - 4*a**2*(u-t)**2 - 1/2*s*a*(u-t)**2 - 1/2*s*(u + t - 4*(a-b))*(u-a)*(t-a) )


check = sp.simplify(sp.expand(0.5*eq_torsten) - sp.expand(eq_start_2))

print(check)