import numpy as np
import vegas 
import matplotlib.pyplot as plt

def f(x):
    return x**2 

# Send arrays in batches
@vegas.batchintegrand
def kernel(x):
    return f(x)

integ = vegas.Integrator([0., 1.])
result1 = integ(kernel, nitn=100, neval=5e5)
result2 = integ(kernel, nitn=100, neval=1e5)
result3 = integ(kernel, nitn=100, neval=5e4)
result4 = integ(kernel, nitn=100, neval=1e4)

true_ans = 1/3

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

r1_hist = [r.mean for r in result1.itn_results]
r2_hist = [r.mean for r in result2.itn_results]
r3_hist = [r.mean for r in result3.itn_results]
r4_hist = [r.mean for r in result4.itn_results]

hist1, bins1, _ = ax1.hist(r1_hist, bins=30, align='left')
hist2, bins2, _ = ax2.hist(r2_hist, bins=30, align='left')
hist3, bins3, _ = ax3.hist(r3_hist, bins=30, align='left')
hist4, bins4, _ = ax4.hist(r4_hist, bins=30, align='left')

ax1.axvline(result1.mean, color='r', ls='--')
ax2.axvline(result2.mean, color='r', ls='--')
ax3.axvline(result3.mean, color='r', ls='--')
ax4.axvline(result4.mean, color='r', ls='--')

ax1.set_title('result1')
ax2.set_title('result2')
ax3.set_title('result3')
ax4.set_title('result4')

fig.tight_layout()
plt.show()


print(result1.mean, abs(true_ans - result1.mean)/true_ans)
print(result2.mean, abs(true_ans - result2.mean)/true_ans)
print(result3.mean, abs(true_ans - result3.mean)/true_ans)
print(result4.mean, abs(true_ans - result4.mean)/true_ans)

