import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pymultinest
import numpy as np
import george
import utils

degree = 8

xfull = np.linspace(0,10,1000)
# Generate scaled polynomial:
yfull = 0
noise = 0.5
for i in range(degree+1):
    coeff = np.random.uniform(-1,1)
    yfull = yfull + (coeff*10**(-i-2))*(xfull**i)
ymean = np.mean(yfull)
yvar = np.var(yfull)
yfull = (yfull-ymean)/np.sqrt(yvar)
yfull_noisy = yfull + np.random.normal(0,noise,len(xfull))
fout = open('data.dat','w')
for i in range(len(yfull)):
    fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(xfull[i],yfull[i],yfull_noisy[i],noise))
plt.plot(xfull,yfull)
plt.errorbar(xfull,yfull_noisy,yerr=np.ones(len(xfull))*noise)
plt.show()
fout.close()
