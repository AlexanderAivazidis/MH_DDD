# Import modules and packages:
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
#%matplotlib inline
sns.set_context('paper')
sns.set_style('darkgrid')
import os
os.chdir('/home/jovyan/MH_DDD/')
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from matplotlib import figure
import pymc3 as pm, theano.tensor as tt
from pymc3.math import logsumexp
#%env THEANO_FLAGS=device=cpu,floatX=float32
import theano
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
from theano.tensor.nlinalg import det
import seaborn as sns
from pymc3Utils import sample_prior, exp_normalize
import pandas as pd
from fnmatch import fnmatch
import pickle
import math as math
import scipy.stats

# Test a 2D Gaussian Mixture Model with 6 components, 5 dimensions and unknown variance and covariances:

## Data:

# Get all evaluation files:
root = '../data/KptnMouse/RNAscope'
pattern = "Objects_Population - Nuclei.txt"
allEvaluationFiles = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
                allEvaluationFiles.append(os.path.join(path, name))
allEvaluationFiles = np.array(allEvaluationFiles)                
# Get all slides that have the cortex, plus cortical depth segmented:
root = 'data/nucleiPositions/'
pattern = "*NucleiNormalizedCorticalDepth.pickle"
allFiles = []
slideNames = []
measurmentNames = []
sectionID = []
evaluationFile = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
            measurmentNames.append(str.split(str.split(str.split(allFiles[-1], '/')[2], '_Nuclei')[0], 'Section')[0])
            slideNames.append(str.split(str.split(str.split(str.split(str.split(allFiles[-1], '/')[2], '_Nuclei')[0], 'Section')[0], '-Measurement')[0], '__')[0])
            sectionID.append(str.split(str.split(str.split(allFiles[-1], '/')[2], '_Nuclei')[0], 'Section')[1])
            evaluationFile.append(allEvaluationFiles[np.array([str.split(allEvaluationFiles[i], '/')[4] for i in range(len(allEvaluationFiles))]) == measurmentNames[-1]][0])

i = 0             
file = open("data/cortexData/" + slideNames[i] + 'Section' + sectionID[i] + 'cortexData_.pickle', 'rb')
cortexData = pickle.load(file)
file.close()

# Use only mid cortex:

cortexData = cortexData[cortexData['Hemisphere'] != 0]

# Run GaussianMixture model:
n_samples = np.shape(cortexData)[0]
data = np.array(cortexData[cortexData.columns[16:21]])
n_dimensions = 5
n_components = 6
alpha = np.array((5,2,1,1,10,5))

# Make some informative prior about mean, variance:
signalMean_priorMean = np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])*0.95)),len(data[:,i]))]) for i in range(n_dimensions)])
backgroundMean_priorMean =  np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])/2)))]) for i in range(n_dimensions)])
signalMean_priorSD = np.array((100,100,100,100,100))
backgroundMean_priorSD = np.array((100,100,100,100,100))

signalMean_priorMean[0] = signalMean_priorMean[0] - 1000

signalSD_priorMean = np.array((100))
signalSD_priorSD = np.array((50))

backgroundSD_priorMean = np.array((50,50,50,50,50))
backgroundSD_priorSD = np.array((50,50,50,50,50))

mus_prior = np.array(((signalMean_priorMean[0], backgroundMean_priorMean[1], backgroundMean_priorMean[2], backgroundMean_priorMean[3], backgroundMean_priorMean[4]), 
                                        (backgroundMean_priorMean[0], signalMean_priorMean[1], backgroundMean_priorMean[2], backgroundMean_priorMean[3], backgroundMean_priorMean[4]),
                                        (backgroundMean_priorMean[0], backgroundMean_priorMean[1], signalMean_priorMean[2], backgroundMean_priorMean[3], backgroundMean_priorMean[4]),
                                        (backgroundMean_priorMean[0], backgroundMean_priorMean[1], backgroundMean_priorMean[2], signalMean_priorMean[3], backgroundMean_priorMean[4]),
                                        (backgroundMean_priorMean[0], backgroundMean_priorMean[1], backgroundMean_priorMean[2], backgroundMean_priorMean[3], signalMean_priorMean[4]),
                                        (backgroundMean_priorMean[0], backgroundMean_priorMean[1], backgroundMean_priorMean[2], backgroundMean_priorMean[3], backgroundMean_priorMean[4])))
sigmas_prior = np.array(((signalMean_priorSD[0], backgroundMean_priorSD[1], backgroundMean_priorSD[2], backgroundMean_priorSD[3], backgroundMean_priorSD[4]), 
                                        (backgroundMean_priorSD[0], signalMean_priorSD[1], backgroundMean_priorSD[2], backgroundMean_priorSD[3], backgroundMean_priorSD[4]),
                                        (backgroundMean_priorSD[0], backgroundMean_priorSD[1], signalMean_priorSD[2], backgroundMean_priorSD[3], backgroundMean_priorSD[4]),
                                        (backgroundMean_priorSD[0], backgroundMean_priorSD[1], backgroundMean_priorSD[2], signalMean_priorSD[3], backgroundMean_priorSD[4]),
                                        (backgroundMean_priorSD[0], backgroundMean_priorSD[1], backgroundMean_priorSD[2], backgroundMean_priorSD[3], signalMean_priorSD[4]),
                                        (backgroundMean_priorSD[0], backgroundMean_priorSD[1], backgroundMean_priorSD[2], backgroundMean_priorSD[3], backgroundMean_priorSD[4]))) 
## Model:

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, taus, n_components):
    def logp_(value):        
        logps = [tt.log(pi[i]) + logp_normal(mus[i,:], taus[i], value) for i in range(n_components)]
        return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))
    return logp_

with pm.Model() as model:
    mus = Normal('mu', mu=pm.floatX(mus_prior), tau=pm.floatX(1/(sigmas_prior**2)), shape=(n_components, n_dimensions))
    pi = Dirichlet('pi', a=pm.floatX(alpha), shape=(n_components,))
    packed_L = [pm.LKJCholeskyCov('packed_L_%d' % i, n=n_dimensions, eta=2., sd_dist=pm.Gamma.dist(mu = signalSD_priorMean, sigma = signalSD_priorSD)) for i in range(n_components)]
    L = [pm.expand_packed_triangular(n_dimensions, packed_L[i]) for i in range(n_components)]
    var = [pm.Deterministic('var_%d' % i, L[i].dot(L[i].T)) for i in range(n_components)]
    taus = [tt.nlinalg.matrix_inverse(var[i]) for i in range(n_components)]
    prior = sample_prior(samples = 1000)
    xs = DensityDist('x', logp_gmix(mus, pi, taus, n_components), observed=data)
    
with model:
    advi_fit = pm.fit(n=10000, obj_optimizer=pm.adagrad(learning_rate=1e-1))  
    
advi_trace = advi_fit.sample(10000)    
advi_summary = pm.summary(advi_trace, include_transformed=False)

advi_elbo = pd.DataFrame(
    {'log-ELBO': -np.log(advi_fit.hist),
     'n': np.arange(advi_fit.hist.shape[0])})
_ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)

pickle_out = open("advi_summary.pickle","wb")
pickle.dump(advi_summary, pickle_out)
pickle_out.close()

## Plot the predicted mean and variance for each component vs. the actual one and also plot the inferred covariance matrices:

# Plot predicted and actual distribution of intensities for each channel:
f, axis = plt.subplots(n_components,n_dimensions, figsize=(n_dimensions*10,n_components*10))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
colours = ('gold', 'pink','green', 'red', 'blue')
x_min = 0
for j in range(n_dimensions):
    x_max = np.max(data[:,j])
    x = np.linspace(x_min, x_max, 100)
    for i in range(n_components):
        axis[i,j].plot(x, scipy.stats.norm.pdf(x,np.mean(advi_trace.get_values('mu')[:,i,j]), np.mean(advi_trace.get_values('sigma_' + str(j))[:,j,j]))*np.mean(advi_trace.get_values('pi')[:,j]), color=colours[j], linewidth = 5)
        axis[i,j].hist(data[:,j], density = True, bins = 50)
# Without weighting my mixture weight:
# Plot predicted and actual distribution of intensities for each channel:

    meanMatrix = np.array(((np.mean(advi_trace.get_values('mu')[:,0,0]),np.mean(advi_trace.get_values('mu')[:,0,1]),np.mean(advi_trace.get_values('mu')[:,0,2]),np.mean(advi_trace.get_values('mu')[:,0,3]),np.mean(advi_trace.get_values('mu')[:,0,4])),
                          (np.mean(advi_trace.get_values('mu')[:,1,0]),np.mean(advi_trace.get_values('mu')[:,1,1]),np.mean(advi_trace.get_values('mu')[:,1,2]),np.mean(advi_trace.get_values('mu')[:,1,3]),np.mean(advi_trace.get_values('mu')[:,1,4])),
                          (np.mean(advi_trace.get_values('mu')[:,2,0]),np.mean(advi_trace.get_values('mu')[:,2,1]),np.mean(advi_trace.get_values('mu')[:,1,2]),np.mean(advi_trace.get_values('mu')[:,2,3]),np.mean(advi_trace.get_values('mu')[:,2,4])),
                          (np.mean(advi_trace.get_values('mu')[:,3,0]),np.mean(advi_trace.get_values('mu')[:,3,1]),np.mean(advi_trace.get_values('mu')[:,1,2]),np.mean(advi_trace.get_values('mu')[:,3,3]),np.mean(advi_trace.get_values('mu')[:,3,4])),
                          (np.mean(advi_trace.get_values('mu')[:,4,0]),np.mean(advi_trace.get_values('mu')[:,4,1]),np.mean(advi_trace.get_values('mu')[:,1,2]),np.mean(advi_trace.get_values('mu')[:,4,3]),np.mean(advi_trace.get_values('mu')[:,4,4])),
                          (np.mean(advi_trace.get_values('mu')[:,5,0]),np.mean(advi_trace.get_values('mu')[:,5,1]),np.mean(advi_trace.get_values('mu')[:,1,2]),np.mean(advi_trace.get_values('mu')[:,5,3]),np.mean(advi_trace.get_values('mu')[:,5,4]))))
    
    sigmaMatrix = np.array(((),
                          (),
                          (),
                          (),
                          (),
                          ()))