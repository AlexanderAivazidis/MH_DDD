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

data = np.array(centralData['Density'])
uniqueAnimals = np.unique(centralData['MouseID'])
animal_Index = [np.where(uniqueAnimals == np.array(centralData['MouseID'])[i])[0][0] for i in range(len(centralData['MouseID']))]
number_uniqueAnimals = len(uniqueAnimals)
genotype_uniqueAnimals = np.array([[np.unique(centralData['Genotype'][centralData['MouseID'] == uniqueAnimals[i]])[0] for i in range(len(uniqueAnimals))][i] == 'WT' for i in range(len(uniqueAnimals))]) * 1

with pm.Model() as model:
    
    alpha = pm.Gamma('Gaussian_Noise_SD', mu = alpha_prior_mean, sd = alpha_prior_sd)
    
    # Prior distribution for animal specific effects: 
    
    a_1 = pm.Gamma('Population_Mean', mu = a_prior_mean, sd = a_prior_sd, shape = 2)
    a_2 = pm.Gamma('Population_SD', mu = a_prior_sd, sd = a_prior_sd/5, shape = 2)
    
    # Animal Specific effect on density with different distribution for each genotype:                
                    
    a = pm.Gamma('Animal_Mean', mu = a_1[genotype_uniqueAnimals], sd = a_2[genotype_uniqueAnimals], shape = number_uniqueAnimals)

    # Linear model with Gaussian noise: 
                    
    x = pm.Normal('x', mu = a[animal_Index], sd = alpha, observed = data)

# Sample:
with model:
    %time hmc_trace = pm.sample(draws=1000, tune=1000, cores=10)

pm.plots.traceplot(hmc_trace, var_names = ['Population_Mean', 'Population_SD'])
pm.plots.plot_posterior(hmc_trace, var_names = ['Population_Mean', 'Population_SD'])
pm.plots.densityplot(hmc_trace, var_names = ['Population_Mean', 'Population_SD'])
axis = pm.plots.forestplot(hmc_trace, kind = 'ridgeplot', combined = True, colors = ('blue', 'red'), ridgeplot_alpha = 1)
axis[1][0].set_title('Hierarchical Gaussian Noise Model \n Parameter Estimates by MCMC')
  
# Plot data alone, with prior predictive distribution and with posterior predictive distribution:

plt.hist(np.array(centralData['Density']), bins = 5)
    