import sympy as sp

def compare_depta(momentum='p3 - p1'):

   if momentum == 'p3 - p1': sgn = 1
   if momentum == 'p1 - p3': sgn = -1

   s, t, a, b = sp.symbols('s t a b')

   """
   a = m_s^2, b = m_phi^2
   """

   # a = 0
   # b = 0 
   u = 2*a + 2*b - s - t

   p1q = -sgn*1/2*(t + a - b)
   p1k = -sgn*1/2*(u + a - b)
   p2q = sgn*1/2*(t + a - b)
   p2k = sgn*1/2*(u + a - b)
   qk = a - b
   p1p2 = 1/2*(s - 2*a)

   """
   From squaring the matrix element 
   iM = -iy^2*cos^4(theta)*v(p1)*[(q + m_s)/(t - a) + (k + m_s)/(u - a)]u(p2)
   and summing (not average) over initial spin d.o.f. we get start_expr 
   (where a factor of 4 is removed):  
   """

   start_expr = ( (u - a)**2*(2*(p1q)*(p2q) - (p1p2)*(t - a) + a*(2*(p1q) - 2*(p2q) - t - a)) 
               + (t - a)**2*(2*(p1k)*(p2k) - (p1p2)*(u - a) + a*(2*(p1k) - 2*(p2k) - u - a))
            + 2*(u-a)*(t-a)*((p1q)*(p2k) + (p1k)*(p2q) - (p1p2)*(qk - a) + a*(p1q - p2q + p1k - p2k - qk - a)) 
   )

   # expression I found for Depta's choice of propagator momentum
   depta = (u-t)**2*(1/2*(a + b - t)*(a + b - u) - 1/2*b*s)

   # expression I found for corrected choice of propagator momentum
   corrected = (u-t)**2*(1/2*(a + b - t)*(a + b - u) - 1/2*b*s) + 4*a*(s - 2*b)*((u - a)*(t + a - b) + (t - a)*(u + a - b))

   check_depta = sp.simplify(start_expr - depta)
   check_corrected = sp.simplify(start_expr - corrected)
   diff_depta_corrected = sp.simplify(depta - corrected)
   print('------------------------------------------')
   print(f'Momentum: {momentum}')
   print(f'Error in Depta: \n{check_depta}\n')
   print(f'Error in corrected: \n{check_corrected}\n')
   print(f'Difference depta - corrected: \n{diff_depta_corrected}\n')
   print('------------------------------------------')

compare_depta(momentum='p1 - p3')
compare_depta(momentum='p3 - p1')