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
    boolean = cortexData['Hemisphere'] != 0
    celltypes = np.zeros(sum(boolean))
    allClasses = np.zeros((sum(boolean),5))
    ### Make final celltype assignements:
    for channel in np.array((1,0,3,4,2)):
        file = open("data/celltypeProbabilities/CortexOnly_" + cortexData['SlideName'][0] + 'Section' + cortexData['Section'][0] + "Classification-" + celltypeOrder[channel] + '.pickle', 'rb')
        normalizedProbs = pickle.load(file)
        file.close()
        maxProbs = [max(normalizedProbs[i]) for i in range(len(normalizedProbs))]
        classMembership = [np.argmax(normalizedProbs[i]) for i in range(len(normalizedProbs))]
        confidentClass = [2 if maxProbs[i] < confidenceThreshold else classMembership[i] for i in range(len(classMembership))]
        allClasses[:,channel] = confidentClass
        celltypes[np.array(confidentClass) == 1] = channel + 1

    ## Remove classifications that are both neuron and astrocyte or opc positive:
    test = [sum(allClasses[i,np.array((0,3,2))]) > 1 or sum(allClasses[i,np.array((0,3,4))]) > 1 for i in range(len(allClasses[:,0]))]
    celltypes[test] = 0

    print(sum(celltypes != 0)/len(celltypes))
    
    ## Save cell type classifications:

    pickle_out = open("data/celltypeClassification/CortexOnly_" + cortexData['SlideName'][0] + 'Section' + cortexData['Section'][0] + "CelltypeClassification.pickle","wb")
    pickle.dump(celltypes, pickle_out)
    pickle_out.close()
    
# Plot cell type classification:

for i in range(len(allFiles)):

    # Import nuclei positions, channel intensities, region classification, cortical depth, radial distance:
    file = open(allFiles[i], 'rb')
    cortexData = pickle.load(file)
    file.close()    
    boolean = cortexData['Hemisphere'] != 0
    
    # Load cell type classification:
    
    file = open("data/celltypeClassification/CortexOnly_" + cortexData['SlideName'][0] + 'Section' + cortexData['Section'][0] + "CelltypeClassification.pickle", 'rb')
    celltypes = pickle.load(file)
    file.close()

    data = np.array(cortexData[cortexData.columns[16:21]])
    figureObjects = ('Nuclei', 'Unclassified', 'Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')
    objectsColours = ('black', 'grey', 'gold', 'pink','green', 'red', 'blue')
    channels = ('568', '490LS', '488', '647', '425')
    colourIndex = np.repeat(int(1), len(celltypes))                            
    for j in range(7):
        colourIndex[np.array(celltypes) == j] = int(j+1)

    f, axis = plt.subplots(2,8, figsize=(40,2*5))
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.facecolor'] = 'white'
    dotSize = 0.5  

    axis[0,0].set_title(figureObjects[0] + ' Positions \n' + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0])
    axis[0,0].scatter(cortexData['x-coordinate'], cortexData['y-coordinate'], s = dotSize, color = objectsColours[0])

    for j in range(6):

        axis[0,j+1].set_title(figureObjects[j+1] + ' Positions \n' + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0])
        axis[0,j+1].scatter(np.array(cortexData['x-coordinate'][boolean])[np.array(celltypes) == j], np.array(cortexData['y-coordinate'][boolean])[np.array(celltypes) == j], s = dotSize, color = objectsColours[j+1])
        if j < 5:
            axis[1,j+2].set_title('Channel ' + channels[j] + ' Intensities and Classification \n' + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0])
            axis[1,j+2].scatter(cortexData['x-coordinate'][boolean], np.log2(data[boolean,j]) ,s = dotSize, c = [objectsColours[colourIndex[i]] for i in range(len(colourIndex))])

    plt.show()
    f.savefig("figures/QC/" + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0] + "Overview.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window