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

# Get all slides that have the cortex, plus cortical depth segmented:
root = 'data/cortexData/'
pattern = "*cortexData_.pickle"
allFiles = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
 
j = 1
columnNames = ('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'Genotype', 'MouseID', 'Hemisphere', 'Total', 'Density')
centralData_Left_subset = pd.DataFrame(index=range(len(allFiles)), columns=columnNames)

for i in range(len(allFiles)):
    print(i)
    file = open(allFiles[i], 'rb')
    cortexData = pickle.load(file)
    file.close()
    points = MultiPoint([(cortexData['x-coordinate'][i], cortexData['y-coordinate'][i]) for i in range(len(cortexData['x-coordinate'])) if cortexData['Hemisphere'][i] == j and cortexData['x_dash-coordinate'][i] < 0.66 and cortexData['x_dash-coordinate'][i] > 0.33])
    centralData_Left_subset['Density'][i] = len(points)/alphashape.alphashape(points, 0.001).area
    centralData_Left_subset['Total'][i] = len(points)
    centralData_Left_subset['SlideName'][i] = cortexData['SlideName'][0]
    centralData_Left_subset['Section'][i] = cortexData['Section'][0]
    centralData_Left_subset['Cycle1-Batch'][i] = cortexData['Cycle1-Batch'][0]
    centralData_Left_subset['Cycle2-Batch'][i] = cortexData['Cycle2-Batch'][0]
    centralData_Left_subset['Genotype'][i] = cortexData['Genotype'][0]
    centralData_Left_subset['MouseID'][i] = cortexData['MouseID'][0]
    centralData_Left_subset['z-coordinate'][i] = cortexData['z-coordinate'][0]
    centralData_Left_subset['Hemisphere'][i] = j

j = 2
columnNames = ('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'Genotype', 'MouseID', 'Hemisphere', 'Total', 'Density')
centralData_Right = pd.DataFrame(index=range(len(allFiles)), columns=columnNames)

for i in range(len(allFiles)):
    print(i)
    file = open(allFiles[i], 'rb')
    cortexData = pickle.load(file)
    file.close()
    points = MultiPoint([(cortexData['x-coordinate'][i], cortexData['y-coordinate'][i]) for i in range(len(cortexData['x-coordinate'])) if cortexData['Hemisphere'][i] == j and cortexData['x_dash-coordinate'][i] < 0.66 and cortexData['x_dash-coordinate'][i] > 0.33])
    centralData_Right['Density'][i] = len(points)/alphashape.alphashape(points, 0.001).area
    centralData_Right['Total'][i] = len(points)
    centralData_Right['SlideName'][i] = cortexData['SlideName'][0]
    centralData_Right['Section'][i] = cortexData['Section'][0]
    centralData_Right['Cycle1-Batch'][i] = cortexData['Cycle1-Batch'][0]
    centralData_Right['Cycle2-Batch'][i] = cortexData['Cycle2-Batch'][0]
    centralData_Right['Genotype'][i] = cortexData['Genotype'][0]
    centralData_Right['MouseID'][i] = cortexData['MouseID'][0]
    centralData_Right['z-coordinate'][i] = cortexData['z-coordinate'][0]
    centralData_Right['Hemisphere'][i] = j

centralData = pd.concat([centralData_Right, centralData_Left_subset])

# Make a plot of total numbers and density, coloured by: Genotype, MouseID, Batch1, Batch2, SectionNumber
# Focus on mid dorsolateral cortical areas:

features = np.array(('Genotype', 'MouseID', 'Cycle1-Batch', 'Cycle2-Batch', 'Hemisphere', 'Section', 'SlideName'))
colours = np.array(('red', 'black', 'yellow', 'green', 'orange', 'pink', 'grey', 'blue', 'red', 'black', 'yellow', 'green', 'orange', 'pink', 'grey'))
markers = np.array(('x', '.', 'v', '^', 's', '*', 'P', '.', 'v', '^', 's', '*', 'P', 'x'))

f, axis = plt.subplots(7,2, figsize=(10,35))
f.subplots_adjust(wspace=0.25, hspace=0.25)

for i in range(len(features)):  
    plotData = centralData[features[i]]
    categories = np.unique(plotData)
    for j in range(len(categories)):
        boolean = plotData == categories[j]
        axis[i,0].scatter(centralData['z-coordinate'][boolean], centralData['Total'][boolean], c = colours[j], marker = markers[j], label = categories[j])
        axis[i,1].scatter(centralData['z-coordinate'][boolean], centralData['Density'][boolean], c = colours[j], marker = markers[j], label = categories[j]) 
    axis[i,1].set_ylim(0,0.005)
    axis[i,1].set_xlabel('z-coordinate')
    axis[i,0].set_xlabel('z-coordinate')
    axis[i,1].set_ylabel('Density ($\mu$m$^{2}$)')
    axis[i,0].set_ylabel('Total Number')
    axis[i,0].set_title('Total Number of Nuclei, Colour: ' + features[i])
    axis[i,1].set_title('Density of Nuclei, Colour: ' + features[i])
    if i != 6:
        axis[i,0].legend()
        axis[i,1].legend()

f.savefig("figures/MidCortexNucleiDensityAndTotal.pdf", bbox_inches='tight')
pickle_out = open("data/nucleiData/centralDataDensityAndTotal.pickle","wb")
pickle.dump(centralData, pickle_out)
pickle_out.close()

# Now make a plot of all sections with left/right hemisphere in red/green, with the relevant data point in the density/total number plot coloured in red/green:

centralData_Left_subset = centralData_Left_subset.sort_values(by=['z-coordinate'])
centralData = centralData.sort_values(by=['z-coordinate'])

size = len(allFiles)
f, axis = plt.subplots(len(allFiles),3, figsize=(15,len(allFiles)*5))
spot_size = 0.05
for j in range(size):
    print(j)
    file = open(allFiles[j], 'rb')
    cortexData = pickle.load(file)
    file.close()
    
    i = np.where([cortexData['SlideName'][0] == np.array(centralData_Left_subset['SlideName'])[i] and cortexData['Section'][0] == np.array(centralData_Left_subset['Section'])[i] for i in range(len(centralData_Left_subset['Section']))])[0][0]
    ii_l = np.where([cortexData['SlideName'][0] == np.array(centralData['SlideName'])[i] and cortexData['Section'][0] == np.array(centralData['Section'])[i] and 1 == np.array(centralData['Hemisphere'])[i] for i in range(len(centralData['Section']))])[0][0]
    ii_r = np.where([cortexData['SlideName'][0] == np.array(centralData['SlideName'])[i] and cortexData['Section'][0] == np.array(centralData['Section'])[i] and 2 == np.array(centralData['Hemisphere'])[i] for i in range(len(centralData['Section']))])[0][0]
    
    binary = cortexData['Hemisphere'] == 1
    axis[i,0].scatter(cortexData['x-coordinate'][binary], cortexData['y-coordinate'][binary], c = 'red', s = spot_size)
    binary = cortexData['Hemisphere'] == 2
    axis[i,0].scatter(cortexData['x-coordinate'][binary], cortexData['y-coordinate'][binary], c = 'blue', s = spot_size)
    binary = cortexData['Hemisphere'] == 0
    axis[i,0].scatter(cortexData['x-coordinate'][binary], cortexData['y-coordinate'][binary], c = 'black', s = spot_size)
    points = MultiPoint([(cortexData['x-coordinate'][i], cortexData['y-coordinate'][i]) for i in range(len(cortexData['x-coordinate'])) if cortexData['Hemisphere'][i] == 1 and cortexData['x_dash-coordinate'][i] < 0.66 and cortexData['x_dash-coordinate'][i] > 0.33])
    boundary = alphashape.alphashape(points, 0.001).boundary.xy
    axis[i,0].plot(boundary[0], boundary[1], c = 'yellow')
    points = MultiPoint([(cortexData['x-coordinate'][i], cortexData['y-coordinate'][i]) for i in range(len(cortexData['x-coordinate'])) if cortexData['Hemisphere'][i] == 2 and cortexData['x_dash-coordinate'][i] < 0.66 and cortexData['x_dash-coordinate'][i] > 0.33])
    boundary = alphashape.alphashape(points, 0.001).boundary.xy
    axis[i,0].plot(boundary[0], boundary[1], c = 'yellow')
    plotData = centralData['Genotype']
    categories = np.unique(plotData)
    colours = np.repeat('black', len(plotData))
    colours[ii_l] = 'red'
    colours[ii_r] = 'blue'
    for j in range(len(categories)):
        boolean = plotData == categories[j]
        subColours = colours[boolean]
        axis[i,1].scatter(centralData['z-coordinate'][boolean], centralData['Total'][boolean], c = subColours, marker = markers[j], label = categories[j])
        axis[i,2].scatter(centralData['z-coordinate'][boolean], centralData['Density'][boolean], c = subColours, marker = markers[j], label = categories[j]) 
    axis[i,2].set_ylim(0,0.005)
    axis[i,2].set_xlabel('z-coordinate')
    axis[i,1].set_xlabel('z-coordinate')
    axis[i,2].set_ylabel('Density ($\mu$m$^{2}$)')
    axis[i,1].set_ylabel('Total Number')
    axis[i,1].set_title('Total Number of Nuclei')
    axis[i,2].set_title('Density of Nuclei')
    axis[i,1].legend()
    axis[i,2].legend()
    axis[i,0].set_title('Slide ' + cortexData['SlideName'][0] + ' Section ' + cortexData['Section'][0])
    
f.savefig("figures/MidCortexNucleiDensityTotalAndAllSections.png", bbox_inches='tight')

# Now make the same plot as before but seperate for each animal:

centralData_Left = centralData_Left.sort_values(by=['z-coordinate'])
centralData = centralData.sort_values(by=['z-coordinate'])

uniqueAnimals = np.unique(centralData['MouseID'])
for k in range(len(uniqueAnimals)):
    
    centralData_Left_subset = centralData_Left[centralData_Left['MouseID'] == uniqueAnimals[k]]
    centralData_subset = centralData[centralData['MouseID'] == uniqueAnimals[k]]

    size = np.shape(centralData_Left_subset)[0]
    f, axis = plt.subplots(size,3, figsize=(15,size*5))
    spot_size = 0.05
    for j in range(len(allFiles)):
        print(j)
        file = open(allFiles[j], 'rb')
        cortexData = pickle.load(file)
        file.close()

        i = np.where([cortexData['SlideName'][0] == np.array(centralData_Left_subset['SlideName'])[i] and cortexData['Section'][0] == np.array(centralData_Left_subset['Section'])[i] for i in range(len(centralData_Left_subset['Section']))])[0]
        ii_l = np.where([cortexData['SlideName'][0] == np.array(centralData_subset['SlideName'])[i] and cortexData['Section'][0] == np.array(centralData_subset['Section'])[i] and 1 == np.array(centralData_subset['Hemisphere'])[i] for i in range(len(centralData_subset['Section']))])[0]
        ii_r = np.where([cortexData['SlideName'][0] == np.array(centralData_subset['SlideName'])[i] and cortexData['Section'][0] == np.array(centralData_subset['Section'])[i] and 2 == np.array(centralData_subset['Hemisphere'])[i] for i in range(len(centralData_subset['Section']))])[0]
        
        if (len(i) > 0):
            
            i = i[0]
            ii_l = ii_l[0]
            ii_r = ii_r[0]
        
            binary = cortexData['Hemisphere'] == 1
            axis[i,0].scatter(cortexData['x-coordinate'][binary], cortexData['y-coordinate'][binary], c = 'red', s = spot_size)
            binary = cortexData['Hemisphere'] == 2
            axis[i,0].scatter(cortexData['x-coordinate'][binary], cortexData['y-coordinate'][binary], c = 'blue', s = spot_size)
            binary = cortexData['Hemisphere'] == 0
            axis[i,0].scatter(cortexData['x-coordinate'][binary], cortexData['y-coordinate'][binary], c = 'black', s = spot_size)
            points = MultiPoint([(cortexData['x-coordinate'][i], cortexData['y-coordinate'][i]) for i in range(len(cortexData['x-coordinate'])) if cortexData['Hemisphere'][i] == 1 and cortexData['x_dash-coordinate'][i] < 0.66 and cortexData['x_dash-coordinate'][i] > 0.33])
            boundary = alphashape.alphashape(points, 0.001).boundary.xy
            axis[i,0].plot(boundary[0], boundary[1], c = 'yellow')
            points = MultiPoint([(cortexData['x-coordinate'][i], cortexData['y-coordinate'][i]) for i in range(len(cortexData['x-coordinate'])) if cortexData['Hemisphere'][i] == 2 and cortexData['x_dash-coordinate'][i] < 0.66 and cortexData['x_dash-coordinate'][i] > 0.33])
            boundary = alphashape.alphashape(points, 0.001).boundary.xy
            axis[i,0].plot(boundary[0], boundary[1], c = 'yellow')
            plotData = centralData_subset['Genotype']
            categories = np.unique(plotData)
            colours = np.repeat('black', len(plotData))
            colours[ii_l] = 'red'
            colours[ii_r] = 'blue'
            for j in range(len(categories)):
                boolean = np.array(plotData) == categories[j]
                subColours = colours[boolean]
                axis[i,1].scatter(centralData_subset['z-coordinate'][boolean], centralData_subset['Total'][boolean], c = subColours, marker = markers[j], label = categories[j])
                axis[i,2].scatter(centralData_subset['z-coordinate'][boolean], centralData_subset['Density'][boolean], c = subColours, marker = markers[j], label = categories[j]) 
            axis[i,2].set_ylim(0,0.005)
            axis[i,2].set_xlabel('z-coordinate')
            axis[i,1].set_xlabel('z-coordinate')
            axis[i,2].set_ylabel('Density ($\mu$m$^{2}$)')
            axis[i,1].set_ylabel('Total Number')
            axis[i,1].set_title('Total Number of Nuclei')
            axis[i,2].set_title('Density of Nuclei')
            axis[i,1].legend()
            axis[i,2].legend()
            axis[i,0].set_title('Slide ' + cortexData['SlideName'][0] + ' Section ' + cortexData['Section'][0])

    f.savefig("figures/MidCortexNucleiDensityTotalAndAllSections_" + uniqueAnimals[k] +".png", bbox_inches='tight')







