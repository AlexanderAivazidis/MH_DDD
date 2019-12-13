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
slideNames = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
            slideNames.append(str.split(str.split(allFiles[-1],'/')[2], '-V')[0])

### The goal is to compare Kptn and WT mouse in terms of total cell numbers and numbers of individual cell types, also relative cell numbers could be worth trying:

metaData = pd.read_excel('data/SectionReferenceSheet.xlsx')

for celltype in np.array(('Neuron', 'GABANeurons', 'Astrocyte', 'OPC')):

    file = open("data/celltypeCounts/CortexOnly_" + celltype + "_CellCounts.pickle", 'rb')
    cellCounts = pickle.load(file)
    file.close()

    file = open("data/celltypeCounts/CortexOnly_Nuclei_CellCounts.pickle", 'rb')
    totalCounts = pickle.load(file)
    file.close()

    colours1 = np.repeat('black', np.shape(cellCounts)[0])
    colours1[cellCounts['Genotype'] == 'Kptn:Hom'] = 'red'
    colours2 = np.repeat('black', np.shape(cellCounts)[0])
    colorSequence = np.array(('red', 'green', 'blue', 'grey', 'black', 'pink'))
    for i in range(len(np.unique(cellCounts['MouseID']))):
        colours2[cellCounts['MouseID'] == np.unique(cellCounts['MouseID'])[i]] = colorSequence[i]
    coords = [np.where([cellCounts['SlideName'][i] == metaData['Automatic SlideID - Cycle 2'][j] and int(cellCounts['Section'][i]) == metaData['Section Number'][j] for j in range(len(metaData['Section Number']))])[0][0] for i in range(len(cellCounts['SlideName']))]

    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Right Cortex'][coords] != 'Broken')
    axis[0].set_title(celltype +  ' Total Numbers')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Cortex'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Left Cortex'][coords] != 'Broken')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Cortex'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Right Cortex'][coords] != 'Broken')    
    axis[1].set_title(celltype + ' Proportions')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Cortex'][boolean]/totalCounts['Right-Cortex'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Left Cortex'][coords] != 'Broken')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Cortex'][boolean]/totalCounts['Left-Cortex'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_Cortex_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window
    
    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Right Cortex'][coords] != 'Broken')
    axis.set_title(celltype +  ' Total Numbers')
    axis.scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Cortex'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Left Cortex'][coords] != 'Broken')
    axis.scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Cortex'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_Cortex_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window
    
    # Also colour by mouse ID:
    
    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Upper Layers Right'][coords] != 'Broken')
    axis[0].set_title(celltype +  ' Total Numbers')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Upper-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Upper Layers Left'][coords] != 'Broken')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Upper-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Upper Layers Right'][coords] != 'Broken')    
    axis[1].set_title(celltype +  ' Proportion')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Upper-Layers'][boolean]/totalCounts['Right-Upper-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Upper Layers Left'][coords] != 'Broken')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Upper-Layers'][boolean]/totalCounts['Left-Upper-Layers'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_UpperLayers_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window

    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Upper Layers Right'][coords] != 'Broken')
    axis[0].set_title(celltype +  ' Total Numbers')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Upper-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Upper Layers Left'][coords] != 'Broken')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Upper-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Upper Layers Right'][coords] != 'Broken')    
    axis[1].set_title(celltype +  ' Proportion')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Upper-Layers'][boolean]/totalCounts['Right-Upper-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Upper Layers Left'][coords] != 'Broken')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Upper-Layers'][boolean]/totalCounts['Left-Upper-Layers'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_UpperLayers_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window

    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Lower Layers Right'][coords] != 'Broken')
    axis[0].set_title(celltype +  ' Total Numbers')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Lower-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Lower Layers Left'][coords] != 'Broken')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Lower-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Lower Layers Right'][coords] != 'Broken')    
    axis[1].set_title(celltype +  ' Proportion')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Lower-Layers'][boolean]/totalCounts['Right-Lower-Layers'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Lower Layers Left'][coords] != 'Broken')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Lower-Layers'][boolean]/totalCounts['Left-Lower-Layers'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_LowerLayers_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window 

    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Medial Right'][coords] != 'Broken')
    axis[0].set_title(celltype +  ' Total Numbers')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Medial'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Medial Left'][coords] != 'Broken')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Medial'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Medial Right'][coords] != 'Broken')    
    axis[1].set_title(celltype +  ' Proportion')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Medial'][boolean]/totalCounts['Right-Medial'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Medial Left'][coords] != 'Broken')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Medial'][boolean]/totalCounts['Left-Medial'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_Medial_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window 

    f, axis = plt.subplots(1,2, figsize=(10,5))
    boolean = np.array(metaData['Lateral Right'][coords] != 'Broken')
    axis[0].set_title(celltype +  ' Total Numbers')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Lateral'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Lateral Left'][coords] != 'Broken')
    axis[0].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Lateral'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Lateral Right'][coords] != 'Broken')    
    axis[1].set_title(celltype +  ' Proportion')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Right-Lateral'][boolean]/totalCounts['Right-Lateral'][boolean], c = colours1[boolean])
    boolean = np.array(metaData['Lateral Left'][coords] != 'Broken')
    axis[1].scatter(cellCounts['z-coordinate'][boolean], cellCounts['Left-Lateral'][boolean]/totalCounts['Left-Lateral'][boolean], c = colours1[boolean])
    plt.show()
    f.savefig("figures/CelltypeProportions/" + celltype + "_Lateral_CelltypeProportions.png", bbox_inches='tight')  
    plt.close(f)    # close the figure window 

# Also plot nuclei numbers:

file = open("data/celltypeCounts/CortexOnly_Nuclei_CellCounts.pickle", 'rb')
totalCounts = pickle.load(file)
file.close()

colours1 = np.repeat('black', np.shape(totalCounts)[0])
colours1[totalCounts['Genotype'] == 'Kptn:Hom'] = 'red'
colours2 = np.repeat('black', np.shape(totalCounts)[0])
colorSequence = np.array(('red', 'green', 'blue', 'grey', 'black', 'pink'))
for i in range(len(np.unique(totalCounts['MouseID']))):
    colours2[totalCounts['MouseID'] == np.unique(totalCounts['MouseID'])[i]] = colorSequence[i]
coords = [np.where([totalCounts['SlideName'][i] == metaData['Automatic SlideID - Cycle 2'][j] and int(totalCounts['Section'][i]) == metaData['Section Number'][j] for j in range(len(metaData['Section Number']))])[0][0] for i in range(len(totalCounts['SlideName']))]

colours3 = np.repeat('black', np.shape(totalCounts)[0])
colours3[totalCounts['Left-Cortex'] == 1] = 'red'

colours3 = np.repeat('black', np.shape(totalCounts)[0])

f, axis = plt.subplots(1,1, figsize=(5,5))
boolean = np.array(metaData['Right Cortex'][coords] != 'Broken')
axis.set_title('Nuclei' +  ' Total Numbers (Red: Kptn:Hom)')
axis.scatter(totalCounts['z-coordinate'][boolean], totalCounts['Right-Cortex'][boolean], c = colours1[boolean])
boolean = np.array(metaData['Left Cortex'][coords] != 'Broken')
axis.scatter(totalCounts['z-coordinate'][boolean], totalCounts['Left-Cortex'][boolean], c = colours1[boolean])
axis.set_xlabel('z-coordinate')
axis.set_ylabel('Total Counts Cortex')
f.savefig("figures/CelltypeProportions/" + 'Nuclei' + "_Cortex_TotalNumbers.png", bbox_inches='tight')  
plt.close(f)    # close the figure window 

f, axis = plt.subplots(1,1, figsize=(5,5))
boolean = np.array(metaData['Right Cortex'][coords] != 'Broken')
axis.set_title('Nuclei' +  ' Total Numbers (Colours: MouseID)')
axis.scatter(totalCounts['z-coordinate'][boolean], totalCounts['Right-Cortex'][boolean], c = colours2[boolean])
boolean = np.array(metaData['Left Cortex'][coords] != 'Broken')
axis.scatter(totalCounts['z-coordinate'][boolean], totalCounts['Left-Cortex'][boolean], c = colours2[boolean])
axis.set_xlabel('z-coordinate')
axis.set_ylabel('Total Counts Cortex')
f.savefig("figures/CelltypeProportions/" + 'Nuclei' + "_Cortex_TotalNumbers_MouseID.png", bbox_inches='tight')  
plt.close(f)    # close the figure window 

f, axis = plt.subplots(1,1, figsize=(5,5))
boolean = np.array(metaData['Right Cortex'][coords] != 'Broken')
axis.set_title('Nuclei' +  ' Total Numbers (Red: Left Cortex)')
axis.scatter(totalCounts['z-coordinate'][boolean], totalCounts['Right-Cortex'][boolean], c = 'black')
boolean = np.array(metaData['Left Cortex'][coords] != 'Broken')
axis.scatter(totalCounts['z-coordinate'][boolean], totalCounts['Left-Cortex'][boolean], c = 'red')
axis.set_xlabel('z-coordinate')
axis.set_ylabel('Total Counts Cortex')
f.savefig("figures/CelltypeProportions/" + 'Nuclei' + "_Cortex_TotalNumbers_Hemisphere.png", bbox_inches='tight')  
plt.close(f)    # close the figure window 
