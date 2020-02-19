# This script just gathers all cell proportions into an array with metadata information and makes some nice plots:

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

# Make a plot cell abundances:

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

celltypeOrder = np.array(('Astrocyte', 'Oligodendrocyte', 'Interneuron', 'OPC', 'ExcitatoryNeuron', 'Other'))
columnNames = np.array(('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'Genotype', 'MouseID', 'Region', 'Hemisphere', 'Astrocyte_mean', 'Oligodendrocyte_mean', 'Interneuron_mean', 'OPC_mean', 'ExcitatoryNeuron_mean','Other_mean', 'Astrocyte_sd', 'Oligodendrocyte_sd', 'GABAergicNeuron_sd', 'OPC_sd', 'ExcitatoryNeuron_sd', 'Other_sd'))

cellTypeProportions = pd.DataFrame(index=range(2*len(slideNames)), columns=columnNames)

for slide in range(len(slideNames)):     
    for hemisphere in np.array((1,2)):
        
        file = open("data/cortexData/" + slideNames[slide] + 'Section' + sectionID[slide] + 'cortexData_.pickle', 'rb')
        cortexData = pickle.load(file)
        file.close()
        
        for key in cellTypeProportions.columns[0:7]:
            cellTypeProportions.loc[slide*2 + (hemisphere -1), key] = cortexData.loc[0,key]
            
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Region'] = 'MidCortex'
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Hemisphere'] = hemisphere
            
        file = open("advi_summaries/advi_summary_slide" + str(slide) + 'hemisphere_' + str(hemisphere) + ".pickle", 'rb')
        advi_summary = pickle.load(file)
        file.close()
        
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Astrocyte_mean'] = advi_summary['mean']['w__0']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Astrocyte_sd'] = advi_summary['sd']['w__0']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Oligodendrocyte_mean'] = advi_summary['mean']['w__1']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Oligodendrocyte_sd'] = advi_summary['sd']['w__1']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Interneuron_mean'] = advi_summary['mean']['w__2']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Interneuron_sd'] = advi_summary['sd']['w__2']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'OPC_mean'] = advi_summary['mean']['w__3']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'OPC_sd'] = advi_summary['sd']['w__3']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'ExcitatoryNeuron_mean'] = advi_summary['mean']['w__4']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'ExcitatoryNeuron_sd'] = advi_summary['sd']['w__4']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Other_mean'] = advi_summary['mean']['w__5']
        cellTypeProportions.loc[slide*2 + (hemisphere -1), 'Other_sd'] = advi_summary['sd']['w__5']
        
        
# Make a plot of density, coloured by: Genotype, MouseID, Batch1, Batch2, SectionNumber

features = np.array(('Genotype', 'MouseID', 'Cycle1-Batch', 'Cycle2-Batch', 'Hemisphere', 'Section', 'SlideName'))
colours = np.array(('red', 'black', 'yellow', 'green', 'orange', 'pink', 'grey', 'blue', 'red', 'black', 'yellow', 'green', 'orange', 'pink', 'grey', 'green', 'orange', 'pink', 'grey', 'blue', 'red', 'black', 'yellow'))
markers = np.array(('x', 'o', 'v', '^', 's', '*', 'P', '.', 'v', '^', 's', '*', 'P', 'x','s', '*', 'P', '.', 'v', '^', 's', '*', '.', 'v', '^', 's', '*', 'x', '.', 'v', '^', 's', '*', 'P', '.', 'v', '^', 's', '*', 'P'))

for celltype in celltypeOrder:
    print(celltype)
    f, axis = plt.subplots(7,1, figsize=(10,35))
    f.subplots_adjust(wspace=0.25, hspace=0.25)

    for i in range(len(features)):  
        plotData = cellTypeProportions[features[i]]
        categories = np.unique(plotData)
        for j in range(len(categories)):
            boolean = plotData == categories[j]
            axis[i].scatter(cellTypeProportions['z-coordinate'][boolean], cellTypeProportions[celltype + '_mean'][boolean], c = colours[j], marker = markers[j], label = categories[j])
    #     axis[i].set_ylim(0,0.005)
        axis[i].set_xlabel('z-coordinate')
        axis[i].set_xlabel('z-coordinate')
        axis[i].set_ylabel('Proportion')
        axis[i].set_ylabel('Total Number')
        axis[i].set_title('Proportion, Colour: ' + features[i])
        axis[i].set_title(celltype + ' Proportion, Colour: ' + features[i])
        if i != 6:
            axis[i].legend()
            axis[i].legend()

    f.savefig("figures/MidCortex" + celltype + "Density.pdf", bbox_inches='tight')
        
# Make a plot with all cell types, coloured by Genotype:

for k in range(len(features)):

    f, axis = plt.subplots(2,3, figsize=(15,10))
    f.subplots_adjust(wspace=0.25, hspace=0.25)
    count = 0

    for i in range(len(celltypeOrder)):
        celltype = celltypeOrder[i]
        plotData = cellTypeProportions[features[k]]
        categories = np.unique(plotData)
        for j in range(len(categories)):
            boolean = plotData == categories[j]
            axis[count % 2, int(np.floor(count/2))].errorbar(cellTypeProportions['z-coordinate'][boolean], cellTypeProportions[celltype + '_mean'][boolean], yerr = cellTypeProportions[celltype + '_sd'][boolean]*1, linestyle = 'None', mfc = colours[j], ecolor = 'grey', marker = markers[j], label = None, ms = 4)
            axis[count % 2, int(np.floor(count/2))].scatter(cellTypeProportions['z-coordinate'][boolean], cellTypeProportions[celltype + '_mean'][boolean], c = colours[j], marker = markers[j], label = categories[j], s = 40)
        axis[count % 2, int(np.floor(count/2))].set_ylim(0,1)
        axis[count % 2, int(np.floor(count/2))].set_xlabel('Rostro-Caudal Coordinate')
        axis[count % 2, int(np.floor(count/2))].set_ylabel('Proportion')
        axis[count % 2, int(np.floor(count/2))].set_title(celltype + ' Proportion')
        axis[count % 2, int(np.floor(count/2))].legend()
        axis[count % 2, int(np.floor(count/2))].legend()
        count += 1

    f.savefig("figures/MidCortex" + celltype + "Density" + "_Colour" + features[k] + ".pdf", bbox_inches='tight')



        






