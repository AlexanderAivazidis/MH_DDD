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
#%env THEANO_FLAGS=device=cpu,floatX=float32,exception_verbosity='high'
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
from scipy.stats import gaussian_kde

# Run a Multivariate Gaussian Mixture Model for mapping the celltypes:

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
n_samples = np.shape(cortexData)[0]
data = np.log2(np.array(cortexData[cortexData.columns[16:21]]))
n_dimensions = 5
n_components = 6

# Plot data:

f, axis = plt.subplots(n_dimensions,n_dimensions, figsize=(n_dimensions*n_dimensions,n_dimensions*n_dimensions))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(n_dimensions):
    for j in range(n_dimensions):
        axis[i,j].scatter(data[:, i], data[:, j], c='g', alpha=0.5)

# Run 

# Make some informative priors:
signalMean_priorMean = np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])*0.95)),len(data[:,i]))]) for i in range(n_dimensions)])
backgroundMean_priorMean =  np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])/2)))]) for i in range(n_dimensions)])
signalMean_priorSD = np.array((0.1,0.1,0.1,0.1,0.1))
backgroundMean_priorSD = np.array((0.01,0.01,0.01,0.01,0.01))
alpha = np.array((5,2,1,1,10,5))

signalSD_priorMean = np.array((1,1,1,1,1))
signalSD_priorSD = np.array((0.5,0.5,0.5,0.5,0.5))

backgroundSD_priorMean = np.array((0.1,0.1,0.1,0.1,0.1))
backgroundSD_priorSD = np.array((0.01,0.01,0.01,0.01,0.01))

componentSD_priorMean = np.array(((signalSD_priorMean[0], backgroundSD_priorMean[1], backgroundSD_priorMean[2], backgroundSD_priorMean[3], backgroundSD_priorMean[4]), 
                                        (backgroundSD_priorMean[0], signalSD_priorMean[1], backgroundSD_priorMean[2], backgroundSD_priorMean[3], backgroundSD_priorMean[4]),
                                        (backgroundSD_priorMean[0], backgroundSD_priorMean[1], signalSD_priorMean[2], backgroundSD_priorMean[3], backgroundSD_priorMean[4]),
                                        (backgroundSD_priorMean[0], backgroundSD_priorMean[1], backgroundSD_priorMean[2], signalSD_priorMean[3], backgroundSD_priorMean[4]),
                                        (backgroundSD_priorMean[0], backgroundSD_priorMean[1], backgroundSD_priorMean[2], backgroundSD_priorMean[3], signalSD_priorMean[4]),
                                        (backgroundSD_priorMean[0], backgroundSD_priorMean[1], backgroundSD_priorMean[2], backgroundSD_priorMean[3], backgroundSD_priorMean[4])))
componentSD_priorSD = np.array(((signalSD_priorSD[0], backgroundSD_priorSD[1], backgroundSD_priorSD[2], backgroundSD_priorSD[3], backgroundSD_priorSD[4]), 
                                        (backgroundSD_priorSD[0], signalSD_priorSD[1], backgroundSD_priorSD[2], backgroundSD_priorSD[3], backgroundSD_priorSD[4]),
                                        (backgroundSD_priorSD[0], backgroundSD_priorSD[1], signalSD_priorSD[2], backgroundSD_priorSD[3], backgroundSD_priorSD[4]),
                                        (backgroundSD_priorSD[0], backgroundSD_priorSD[1], backgroundSD_priorSD[2], signalSD_priorSD[3], backgroundSD_priorSD[4]),
                                        (backgroundSD_priorSD[0], backgroundSD_priorSD[1], backgroundSD_priorSD[2], backgroundSD_priorSD[3], signalSD_priorSD[4]),
                                        (backgroundSD_priorSD[0], backgroundSD_priorSD[1], backgroundSD_priorSD[2], backgroundSD_priorSD[3], backgroundSD_priorSD[4])))

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

# Sparse model:
with pm.Model() as model:
    # Weights of each component:
    w = Dirichlet('w', a=pm.floatX(alpha), shape=(n_components,))
    
    # Impose structure onto mean and standard deviation with off-diagonal elements all being the same, because background should be the same throughout:
    mus_signal = MvNormal('mus_signal', mu=pm.floatX(signalMean_priorMean), tau=pm.floatX(np.eye(n_dimensions)/signalMean_priorSD**2), shape=n_dimensions)
    mus_background = MvNormal('mus_background', mu=pm.floatX(backgroundMean_priorMean), tau=pm.floatX(np.eye(n_dimensions)/backgroundMean_priorSD**2), shape=n_dimensions)
    mus = tt.fill_diagonal(tt.reshape(tt.tile(mus_background, n_components), (6,5)),0) + tt.eye(n_components, n_dimensions) * mus_signal
    sigmas_signal = pm.Gamma('sigmas_signal', mu=pm.floatX(signalSD_priorMean), sd=pm.floatX(signalSD_priorSD), shape=n_dimensions)
    sigmas_background = pm.Gamma('sigmas_background', mu=pm.floatX(backgroundSD_priorMean), sd=pm.floatX(backgroundSD_priorSD), shape=n_dimensions)
    sigmas = tt.fill_diagonal(tt.reshape(tt.tile(sigmas_background, n_components), (6,5)),0) + tt.eye(n_components, n_dimensions) * sigmas_signal
    
    # Now get prior for covariance matrix with strong prior on standard deviation from above:
    packed_L = [pm.LKJCholeskyCov('packed_L_%d' % i, n=n_dimensions, eta=1., sd_dist=pm.Gamma.dist(mu = sigmas[i,:], sd = 10**(-25))) for i in range(n_components)]
    L = [pm.expand_packed_triangular(n_dimensions, packed_L[i]) for i in range(n_components)]              
    covs = [pm.Deterministic('cov_%d' % i, tt.dot(L[i],L[i].T)) for i in range(n_components)]   
    taus = [tt.nlinalg.matrix_inverse(covs[i]) for i in range(n_components)]
    
    # Gaussian Mixture Model:
    xs = DensityDist('x', logp_gmix(mus, w, taus, n_components), observed=pm.floatX(data))
    
with model:
    advi_fit = pm.fit(n=25000, obj_optimizer=pm.adagrad(learning_rate=1e-1))  
    
advi_trace = advi_fit.sample(10000)    
advi_summary = pm.summary(advi_trace, include_transformed=False)

advi_elbo = pd.DataFrame(
    {'log-ELBO': -np.log(advi_fit.hist),
     'n': np.arange(advi_fit.hist.shape[0])})
_ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
plt.savefig('books_read.png')

pickle_out = open("advi_summary1.pickle","wb")
pickle.dump(advi_summary, pickle_out)
pickle_out.close()

## Plot the predicted mean and variance for each component vs. the actual one and also plot the inferred covariance matrices:

meanMatrix = np.array(((np.mean(advi_trace.get_values('mus_signal')[:,0]),np.mean(advi_trace.get_values('mus_background')[:,1]),np.mean(advi_trace.get_values('mus_background')[:,2]),np.mean(advi_trace.get_values('mus_background')[:,3]),np.mean(advi_trace.get_values('mus_background')[:,4])),
                      (np.mean(advi_trace.get_values('mus_background')[:,0]),np.mean(advi_trace.get_values('mus_signal')[:,1]),np.mean(advi_trace.get_values('mus_background')[:,2]),np.mean(advi_trace.get_values('mus_background')[:,3]),np.mean(advi_trace.get_values('mus_background')[:,4])),
                      (np.mean(advi_trace.get_values('mus_background')[:,0]),np.mean(advi_trace.get_values('mus_background')[:,1]),np.mean(advi_trace.get_values('mus_signal')[:,2]),np.mean(advi_trace.get_values('mus_background')[:,3]),np.mean(advi_trace.get_values('mus_background')[:,4])),
                      (np.mean(advi_trace.get_values('mus_background')[:,0]),np.mean(advi_trace.get_values('mus_background')[:,1]),np.mean(advi_trace.get_values('mus_background')[:,2]),np.mean(advi_trace.get_values('mus_signal')[:,3]),np.mean(advi_trace.get_values('mus_background')[:,4])),
                      (np.mean(advi_trace.get_values('mus_background')[:,0]),np.mean(advi_trace.get_values('mus_background')[:,1]),np.mean(advi_trace.get_values('mus_background')[:,2]),np.mean(advi_trace.get_values('mus_background')[:,3]),np.mean(advi_trace.get_values('mus_signal')[:,4])),
                      (np.mean(advi_trace.get_values('mus_background')[:,0]),np.mean(advi_trace.get_values('mus_background')[:,1]),np.mean(advi_trace.get_values('mus_background')[:,2]),np.mean(advi_trace.get_values('mus_background')[:,3]),np.mean(advi_trace.get_values('mus_background')[:,4]))))

sigmaMatrix = [[],[],[],[],[],[]]
for i in range(n_components):
    sigmaMatrix[i] = np.empty((n_dimensions,n_dimensions))
    for j in range(n_dimensions):
        for k in range(n_dimensions):
            sigmaMatrix[i][j,k] = np.mean(advi_trace.get_values('cov_' + str(i))[:,j,k])
    
weightsVector = np.array((np.mean(advi_trace.get_values('w')[:,0]), np.mean(advi_trace.get_values('w')[:,1]),np.mean(advi_trace.get_values('w')[:,2]),np.mean(advi_trace.get_values('w')[:,3]),np.mean(advi_trace.get_values('w')[:,4]),np.mean(advi_trace.get_values('w')[:,5]),))

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
        if i == j:
            axis[i,j].plot(x, scipy.stats.norm.pdf(x,meanMatrix[i,j], sigmaMatrix[i][j,j]), color=colours[j], linewidth = 5)
        if i != j:
            axis[i,j].plot(x, scipy.stats.norm.pdf(x,meanMatrix[i,j], sigmaMatrix[i][j,j]), color=colours[j], linewidth = 5)
        axis[i,j].hist(data[:,j], density = True, bins = 50)       
        
# Log likelihood of normal distribution
def logp_normal_np(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * np.log(2 * np.pi) + np.log(1./np.linalg.det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))
        
f, axis = plt.subplots(n_dimensions,n_dimensions, figsize=(n_dimensions*n_dimensions,n_dimensions*n_dimensions))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(n_dimensions):
    for j in range(n_dimensions):
        if i != j:
            xy = np.vstack([data[:,i],data[:,j]])
            z = gaussian_kde(xy)(xy)
            X, Y = np.meshgrid(np.linspace(np.min(data[:, i]), np.max(data[:, i])), np.linspace(np.min(data[:, j]),np.max(data[:, j])))
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z = logp_normal_np(meanMatrix[i,(i,j)], np.linalg.inv(sigmaMatrix[i][[[i,j], [[i],[j]]]])**2, XX)
            Z = np.exp(Z.reshape((50,50)))
            axis[i,j].scatter(data[:, i], data[:, j], c=z, s=50, edgecolor='')
            axis[i,j].scatter(meanMatrix[0, i], meanMatrix[0, j], c='red', s=100)
            axis[i,j].scatter(meanMatrix[1, i], meanMatrix[1, j], c='blue', s=100)
            axis[i,j].scatter(meanMatrix[2, i], meanMatrix[2, j], c='orange', s=100)
            axis[i,j].scatter(meanMatrix[3, i], meanMatrix[3, j], c='black', s=100)
            axis[i,j].scatter(meanMatrix[4, i], meanMatrix[4, j], c='pink', s=100)
            axis[i,j].scatter(meanMatrix[5, i], meanMatrix[5, j], c='purple', s=100)
            axis[i,j].contour(X, Y, Z, levels = (0.0001, 0.001))
