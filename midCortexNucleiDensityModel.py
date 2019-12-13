### Focus analysis on dorsolateral cortex and also look at density rather than just total number

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
from pymc3Distributions import logp_normal
import pandas as pd
from fnmatch import fnmatch
import pickle
import math as math
import scipy.stats
from shapely.geometry import MultiPoint, Polygon
from descartes import PolygonPatch
import alphashape

file = open("data/nucleiData/centralDataDensityAndTotal.pickle", 'rb')
centralData = pickle.load(file)
file.close()

a_prior_mean = np.mean(centralData['Density'])
a_prior_sd = np.sqrt(np.var(centralData['Density']))/2
alpha_prior_mean = np.sqrt(np.var(centralData['Density']))
alpha_prior_sd = np.sqrt(np.var(centralData['Density']))/2

data = centralData['Density']

with pm.Model() as model:
    a = pm.Gamma('a', mu = a_prior_mean, sd = a_prior_sd)
    alpha = pm.Gamma('alpha', mu = alpha_prior_mean, sd = alpha_prior_sd)
    mean_density = a
    sd_density = alpha
    x = pm.Normal('x', mu = mean_density, sd = sd_density, observed = data)

# Sample:
with model:
    %time hmc_trace = pm.sample(draws=1000, tune=10000, cores=10)
    
    