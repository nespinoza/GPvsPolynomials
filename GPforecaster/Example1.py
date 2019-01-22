import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pymultinest
import numpy as np
import george
import utils

degree = 8

xfull,yfull,ynoisy,noise = np.loadtxt('data.dat',unpack=True)
#n = 10
#idx = list(np.arange(0,1000,1000./n).astype('int'))#[100,300,500,700,900]
idx = list(np.arange(350,750,10).astype('int'))
print idx
x = xfull[idx]
y = ynoisy[idx]
yerr = noise[idx]
# Fit GP:
kernel = 1.*george.kernels.ExpSquaredKernel(metric = 100.0)
gp = george.GP(kernel, mean=0.0,fit_mean=False,solver=george.HODLRSolver)
gp.compute(x,yerr)
def prior(cube, ndim, nparams):
    # Prior on GP lamplitude:
    cube[0] = utils.transform_uniform(cube[0],-10,10)
    # Prior on GP llengthscale:
    cube[1] = utils.transform_uniform(cube[1],-10,10)
    

def loglike(cube, ndim, nparams):
    # Extract parameters:
    a,l = cube[0],cube[1]
    # Generate model:
    gp.set_parameter_vector([a,l])
    # Evaluate the log-likelihood:
    loglikelihood = gp.log_likelihood(y)
    return loglikelihood

n_params = 2
out_file = 'out_multinest'
# Run MultiNest:
pymultinest.run(loglike, prior, n_params, n_live_points = 500,outputfiles_basename=out_file, resume = False, verbose = True)
output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
#plt.plot(posterior_samples[:,0],posterior_samples[:,1],'.')
#plt.show()
#sys.exit()
sns.set_context("talk")
sns.set_style("ticks")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = '5' 
matplotlib.rcParams['axes.linewidth'] = 1.2 
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['lines.markeredgewidth'] = 1

# Train GP on points:
meangp = np.array([])
nsamples = 300
idx = np.random.choice(np.arange(len(posterior_samples[:,0])),nsamples,replace=False)
for i in range(nsamples):
    a,l = posterior_samples[idx[i],0],posterior_samples[idx[i],0]
    gp.set_parameter_vector([a,l])
    ygp = gp.sample_conditional(y,xfull)
    if i == 0:
        meangp = ygp
    else:  
        meangp = np.vstack((meangp,ygp))
    plt.plot(xfull,ygp,color='black',linewidth=1,alpha=0.05)
ms = np.zeros(len(xfull))
ms1up,ms1down = np.zeros(len(xfull)),np.zeros(len(xfull))
ms2up,ms2down = np.zeros(len(xfull)),np.zeros(len(xfull))
ms3up,ms3down = np.zeros(len(xfull)),np.zeros(len(xfull))
for i in range(len(xfull)):
    m,s1u,s1d = utils.get_quantiles(meangp[:,i])
    m,s2u,s2d = utils.get_quantiles(meangp[:,i],alpha=0.95)
    #m,s3u,s3d = utils.get_quantiles(meangp[:,i],alpha=0.99)
    ms[i] = m
    ms1up[i],ms1down[i] = s1u,s1d
    ms2up[i],ms2down[i] = s2u,s2d
    #ms3up[i],ms3down[i] = s3u,s3d
plt.fill_between(xfull,ms1down,ms1up,color='royalblue',alpha=0.8)
plt.fill_between(xfull,ms2down,ms2up,color='cornflowerblue',alpha=0.8)
plt.plot(xfull,ms,color='black',label='Posterior GP')
# Plot it:
plt.errorbar(x,y,yerr=yerr,fmt='o',markersize=5,mfc='white',mec='black',ecolor='black',elinewidth=1)
plt.plot(xfull,yfull,label='Real model',color='red')
coeff = np.polyfit(x,y,degree)
plt.plot(xfull,np.polyval(coeff,xfull),label='Estimated polynomial',color='orange')
plt.xlabel('Time index')
plt.ylabel('Amplitude index')
plt.legend()
plt.xlim([0,10])
plt.ylim([-4,4])
plt.savefig('plot.png',dpi=300)
