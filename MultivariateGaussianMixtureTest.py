# Import modules and packages:
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
#%matplotlib inline
sns.set_context('paper')
sns.set_style('darkgrid')
import os
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from matplotlib import figure
import pymc3 as pm, theano.tensor as tt
from pymc3.math import logsumexp
#%env THEANO_FLAGS=device=cuda,floatX=float32
import theano
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
from theano.tensor.nlinalg import det
import seaborn as sns
from pymc3Utils import sample_prior, exp_normalize
from pymc3Distributions import logp_normal
import pandas as pd
from fnmatch import fnmatch
import pickle
import math as math
import scipy.stats

# Test a 2D Gaussian Mixture Model with 6 components, 5 dimensions and unknown variance and covariances:

n_samples = 30000
n_components = 6
n_dimensions = 5
ms = np.array(((10,7,8,7,6),(8,9,7,7,6.5),(8,7,9,7,6.5),(8,7,9,10,6.5),(8,7,9,6.5,8),(8,7,9,6.5,6)))
sds = np.array(((2,0.5,0.5,0.5,0.3),(0.1,1,0.2,0.2,0.2),(0.1,0.1,1,0.1,0.1),(0.2,0.1,0.1,1,0.2),(0.1,0.1,0.1,0.3,1),(0.1,0.1,0.1,0.1,0.1)))
correlationMatrix = np.array((np.array(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1))),
                            np.array(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1))),
                            np.array(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1))),
                            np.array(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1))),
                            np.array(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1))),
                            np.array(((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1)))))                             
cov = [correlationMatrix[i]*sds[i] for i in range(len(sds))]
rng = np.random.RandomState(123)
ps = np.array([0.05, 0.05, 0.2, 0.3, 0.1, 0.2])

zs = [np.where(rng.multinomial(1, ps) != 0)[0][0] for _ in range(n_samples)]
data = np.array([rng.multivariate_normal(ms[i], cov[i], size=1)[0] for i in zs])

f, axis = plt.subplots(n_dimensions,n_dimensions, figsize=(n_dimensions*n_dimensions,n_dimensions*n_dimensions))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(n_dimensions):
    for j in range(n_dimensions):
        axis[i,j].scatter(data[:, i], data[:, j], c='g', alpha=0.5)
        axis[i,j].scatter(ms[0, i], ms[0, j], c='red', s=100)
        axis[i,j].scatter(ms[1, i], ms[1, j], c='blue', s=100)
        axis[i,j].scatter(ms[2, i], ms[2, j], c='orange', s=100)
        axis[i,j].scatter(ms[3, i], ms[3, j], c='black', s=100)
        axis[i,j].scatter(ms[4, i], ms[4, j], c='pink', s=100)
        axis[i,j].scatter(ms[5, i], ms[5, j], c='purple', s=100)

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

## Prior for model:

componentMean = ms + np.random.uniform(0,5,n_dimensions)
componentTau = np.random.uniform(0,2,n_dimensions) * np.eye(n_dimensions)

with pm.Model() as model:
    mus = MvNormal('mu', mu=pm.floatX(componentMean), tau=pm.floatX(componentTau), shape=(n_components, n_dimensions))
    pi = Dirichlet('pi', a=pm.floatX(0.1 * np.ones(n_components)), shape=(n_components,))
    packed_L = [pm.LKJCholeskyCov('packed_L_%d' % i, n=n_dimensions, eta=2., sd_dist=pm.HalfCauchy.dist(2.5)) for i in range(n_components)]
    L = [pm.expand_packed_triangular(n_dimensions, packed_L[i]) for i in range(n_components)]
    sigmas = [pm.Deterministic('sigma_%d' % i, L[i].dot(L[i].T)) for i in range(n_components)]
    taus = [tt.nlinalg.matrix_inverse(sigmas[i]) for i in range(n_components)]
    xs = DensityDist('x', logp_gmix(mus, pi, taus, n_components), observed=data)
    
with model:
    advi_fit = pm.fit(n=500, obj_optimizer=pm.adagrad(learning_rate=1e-1))  
    
advi_trace = advi_fit.sample(10000)    
advi_summary = pm.summary(advi_trace)

pickle_out = open("advi_summary.pickle","wb")
pickle.dump(advi_summary, pickle_out)
pickle_out.close()
