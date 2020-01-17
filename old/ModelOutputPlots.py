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
import sys

# Get all slides that have the cortex, plus cortical depth segmented:
root = 'data/cortexData/'
pattern = "*cortexData_.pickle"
allFiles = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))   

celltypeOrder = np.array(('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron'))
for i in range(len(allFiles)):

    # Import nuclei positions, channel intensities, region classification, cortical depth, radial distance:
    file = open(allFiles[i], 'rb')
    cortexData = pickle.load(file)
    file.close()            

    confidenceThreshold = 0.99
    celltypes = np.zeros(sum(cortexData['Hemisphere'] != 0))
    allClasses = np.zeros((sum(cortexData['Hemisphere'] != 0),5))
    ### Make final celltype assignements:
    for channel in np.array((1,0,3,4,2)):
        file = open("data/celltypeClassification/CortexOnly_" + cortexData['SlideName'][0] + 'Section' + cortexData['Section'][0] + "Classification-" + celltypeOrder[channel] + '.pickle', 'rb')
        normalizedProbs = pickle.load(file)
        file.close()
        maxProbs = [max(normalizedProbs[i]) for i in range(len(normalizedProbs))]
        classMembership = [np.argmax(normalizedProbs[i]) for i in range(len(normalizedProbs))]
        confidentClass = [2 if maxProbs[i] < confidenceThreshold else classMembership[i] for i in range(len(classMembership))]
        allClasses[:,channel] = confidentClass
        celltypes[np.array(confidentClass) == 1] = channel + 1

    ## Remove classifications that are both neuron and astrocyte or opc positive:
    test = [sum(allClasses[i,np.array((1,2,3))]) > 1 or sum(allClasses[i,np.array((1,2,4))]) > 1 for i in range(len(allClasses[:,0]))]
    celltypes[test] = 0

    print(sum(celltypes != 0)/len(celltypes))
