import os
os.chdir('/home/jovyan/MH_DDD/')
from fnmatch import fnmatch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')

### The goal is to compare Kptn and WT mouse in terms of total cell numbers and 

# Load cortexData and cell classification:

# Get all slides that have the cortex, plus cortical depth segmented:
root = 'data/cortexData/'
pattern = "*cortexData_.pickle"
allFiles = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))   

# Import metadata to know which sections/parts to exclude:

metaData = pd.read_excel('data/SectionReferenceSheet.xlsx')

# Make a dataframe with each section name and metadata, plus number of cells of each cell type in all parts of the section. 

columnNames = ('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'Genotype', 'MouseID', 'Cortex', 'Left-Cortex', 'Right-Cortex', 'Left-Upper-Layers', 'Right-Upper-Layers', 'Left-Lower-Layers', 'Right-Lower-Layers', 'Left-Medial', 'Right-Medial', 'Left-Lateral', 'Right-Lateral')
Neuron_Data = pd.DataFrame(index=range(len(allFiles)), columns=columnNames)
for i in range(len(allFiles)):
    print(i)
    file = open(allFiles[i], 'rb')
    cortexData = pickle.load(file)
    file.close()
    boolean = (cortexData['Hemisphere'] != 0)
    cortexData = cortexData[boolean]
    file = open("data/celltypeClassification/CortexOnly_" + np.array(cortexData['SlideName'])[0] + 'Section' + np.array(cortexData['Section'])[0] + "CelltypeClassification.pickle", 'rb')
    celltypes = pickle.load(file)
    file.close()
    Neuron_Data['SlideName'][i] = np.array(cortexData['SlideName'])[0]
    Neuron_Data['Section'][i] = np.array(cortexData['Section'])[0]
    Neuron_Data['Cycle1-Batch'][i] = np.array(cortexData['Cycle1-Batch'])[0]
    Neuron_Data['Cycle2-Batch'][i] = np.array(cortexData['Cycle2-Batch'])[0]
    Neuron_Data['z-coordinate'][i] = np.array(cortexData['z-coordinate'])[0]
    Neuron_Data['Genotype'][i] = np.array(cortexData['Genotype'])[0]
    Neuron_Data['MouseID'][i] = np.array(cortexData['MouseID'])[0]
    Hemisphere = np.array(cortexData['Hemisphere'])
    UpperLower = np.array(cortexData['Upper-Lower'])
    MedialLateral = np.array(cortexData['Medial-Lateral'])
    Neuron_Data['Left-Cortex'][i] = sum([Hemisphere[j] == 1 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Right-Cortex'][i] = sum([Hemisphere[j] == 2 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Cortex'][i] = sum([Hemisphere[j] in np.array((1,2)) and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Left-Upper-Layers'][i] = sum([Hemisphere[j] == 1 and UpperLower[j] == 1 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Right-Upper-Layers'][i] = sum([Hemisphere[j] == 2 and UpperLower[j] == 1 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Left-Lower-Layers'][i] = sum([Hemisphere[j] == 1 and UpperLower[j] == 0 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Right-Lower-Layers'][i] = sum([Hemisphere[j] == 2 and UpperLower[j] == 0 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Left-Medial'][i] = sum([Hemisphere[j] == 1 and MedialLateral[j] == 1 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Right-Medial'][i] = sum([Hemisphere[j] == 2 and MedialLateral[j] == 1 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Left-Lateral'][i] = sum([Hemisphere[j] == 1 and MedialLateral[j] == 0 and celltypes[j] == 5 for j in range(len(celltypes))])
    Neuron_Data['Right-Lateral'][i] = sum([Hemisphere[j] == 2 and MedialLateral[j] == 0 and celltypes[j] == 5 for j in range(len(celltypes))])
    
pickle_out = open("data/celltypeCounts/CortexOnly_Neuron_CellCounts.pickle", 'wb')
pickle.dump(Neuron_Data, pickle_out)
pickle_out.close()