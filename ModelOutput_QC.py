import os
os.chdir('/home/jovyan/MH_DDD/')
from fnmatch import fnmatch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')

### The goal is to check the output of the Gaussian Mixture model:

# Get all evaluation files:
root = '../data/KptnMouse/RNAscope'
pattern = "Objects_Population - Nuclei.txt"
allFiles = []
measurementNames = []
slideNames = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
            measurementNames.append(str.split(allFiles[-1], '/')[4])
            slideNames.append(str.split(measurementNames[-1],'__')[0])

# Import corresponding batch and figure numbers:

metaData = pd.read_excel('data/SectionReferenceSheet.xlsx')

## Load the mean and variances of all components, plus probabilities for each cell type:

columnNames = ('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'mu_0', 'sigma_0', 'w_0', 'mu_1', 'sigma_1', 'w_1')
mixtureOutput = pd.DataFrame(index=range(np.shape(metaData)[0]), columns=columnNames)
mixtureOutput['SlideName'] = [metaData['Automatic SlideID - Cycle 2'][i] if str(metaData['Automatic SlideID - Cycle 2'][i]) != 'nan' else metaData['SlideID - Cycle 2'][i] for i in range(np.shape(metaData)[0])]
mixtureOutput['Section'] = metaData['Section Number']
mixtureOutput['Cycle1-Batch'] = metaData['Batch - Cycle 1']
mixtureOutput['Cycle2-Batch'] = metaData['Batch -- Cycle 2']
mixtureOutput['z-coordinate'] = metaData['Figure Number - Sectioning']
mixtureOutputList = [pd.DataFrame.copy(mixtureOutput), pd.DataFrame.copy(mixtureOutput), pd.DataFrame.copy(mixtureOutput), pd.DataFrame.copy(mixtureOutput), pd.DataFrame.copy(mixtureOutput)]
# for i in range(np.shape(mixtureOutput)[0]):
    
#     for channel in range(5):
        
#         file = open("data/" + measurementNames[int(np.where(np.array(slideNames) == mixtureOutputList[channel]['SlideName'][i])[0])] + 'Section' + str(mixtureOutputList[channel]['Section'][i]) + 'Channel' + str(channel) +'_AdviFitResults.pickle', 'rb')
#         data = pickle.load(file)
#         file.close()
        
#         mixtureOutputList[channel]['mu_0'][i] = data['advi_mu_0']
#         mixtureOutputList[channel]['mu_1'][i] = data['advi_mu_1']
#         mixtureOutputList[channel]['sigma_0'][i] = data['advi_sigma_0']
#         mixtureOutputList[channel]['sigma_1'][i] = data['advi_sigma_1']
#         mixtureOutputList[channel]['w_0'][i] = data['advi_w_0']
#         mixtureOutputList[channel]['w_1'][i] = data['advi_w_1']

#  # Let's sort according to z coordinate:
for i in range(len(mixtureOutputList)):
    mixtureOutputList[i] = mixtureOutputList[i].sort_values('z-coordinate')       
        
# Obtain final cell type assignements, by removing cells with multiple assignements:

columnNames = ('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron', 'Unclassified')
cellTypeAssignments = pd.DataFrame(index=range(np.shape(metaData)[0]), columns=columnNames)
cellTypeAssignments['SlideName'] = [metaData['Automatic SlideID - Cycle 2'][i] if str(metaData['Automatic SlideID - Cycle 2'][i]) != 'nan' else metaData['SlideID - Cycle 2'][i] for i in range(np.shape(metaData)[0])]
cellTypeAssignments['Section'] = metaData['Section Number']
cellTypeAssignments['Cycle1-Batch'] = metaData['Batch - Cycle 1']
cellTypeAssignments['Cycle2-Batch'] = metaData['Batch -- Cycle 2']
cellTypeAssignments['z-coordinate'] = metaData['Figure Number - Sectioning']
celltypeOrder = ('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')

for i in range(np.shape(mixtureOutput)[0]):
        
        print(i)
        assignment_probs = []
        for channel in range(5):
            
            if os.path.exists("data/" + measurementNames[int(np.where(np.array(slideNames) == mixtureOutputList[channel]['SlideName'][i])[0])] + 'Section' + str(mixtureOutputList[channel]['Section'][i]) + "Probability-" + celltypeOrder[channel] + '.pickle'):
                    file = open("data/" + measurementNames[int(np.where(np.array(slideNames) == mixtureOutputList[channel]['SlideName'][i])[0])] + 'Section' + str(mixtureOutputList[channel]['Section'][i]) + "Probability-" + celltypeOrder[channel] + '.pickle', 'rb')
                    assignment_probs.append(pickle.load(file))
                    file.close()                                    
            else:
                print('This file does not exist:' + "data/" + measurementNames[int(np.where(np.array(slideNames) == mixtureOutputList[channel]['SlideName'][i])[0])]+ 'Section' + str(mixtureOutputList[channel]['Section'][i]) + "Probability-" + celltypeOrder[channel] + '.pickle')
        
        if len(assignment_probs) > 0:
            assignment_probs = np.array(np.array(assignment_probs)[:,:,1])
            assignment = 1*(assignment_probs > 0.95)
            combinedAssignements = [np.where(assignment[:,i] == 1)[0] + 1 if len(np.where(assignment[:,i] == 1)[0]) > 0 else np.array([0])for i in range(np.shape(assignment)[1])] 
            # Filter out doublets, except GABAergic neurons which can be both GABA and NeuN positive, also Neurons can be Plp1 positive due to bleedthrough.
            finalAssignement = np.zeros(len(combinedAssignements))
            for j in range(len(combinedAssignements)):
                if np.all(np.sort(combinedAssignements[j]) == np.sort(np.array([3,5]))):
                    finalAssignement[j] = 3
                if len(combinedAssignements[j]) > 1:
                    finalAssignement[j] = 0
                else:
                    finalAssignement[j] = combinedAssignements[j][0]
            
            cellTypeAssignments['Unclassified'][i] = sum(finalAssignement == 0)
            cellTypeAssignments['Astrocyte'][i] = sum(finalAssignement == 1)
            cellTypeAssignments['Oligodendrocyte'][i] = sum(finalAssignement == 2)
            cellTypeAssignments['GABAergicNeuron'][i] = sum(finalAssignement == 3)
            cellTypeAssignments['OPC'][i] = sum(finalAssignement == 4)
            cellTypeAssignments['Neuron'][i] = sum(finalAssignement == 5)
            
            pickle_out = open("data/" + mixtureOutputList[channel]['SlideName'][i] + 'Section' + str(mixtureOutputList[channel]['Section'][i]) + '_FinalCelltypeAssignments.pickle',"wb")
            pickle.dump(finalAssignement, pickle_out)
            pickle_out.close()
        
## Make some plots of cell type proportions and w, mean and sd's of each component + sample mean to check for outliers, together with batch number and section z position.

# Mean and sd of positive component for each channel:

#Mean:
figureObjects = ('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')
channels = ('568', '490LS', '488', '647', '425')
colours = np.repeat('black', len(metaData['Genotype']))                            
colours[np.array(metaData['Genotype']) == 'Kptn:Hom'] = 'red'  
f, axis = plt.subplots(5,1, figsize=(5,25))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(5):   
    axis[i].scatter(mixtureOutputList[0]['z-coordinate'], mixtureOutputList[i]['mu_1'], color = colours)
    plt.xlabel('z-coordinate')
    plt.ylabel('estimated mean intensity')
    axis[i].set_title(figureObjects[i] + '\n Estimated Mean Intensity Channel ' + channels[i])

plt.show()
f.savefig("figures/QC/ComponentMeanVsZposition.png", bbox_inches='tight')  
plt.close(f)    # close the figure window
    
#sd:
figureObjects = ('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')
channels = ('568', '490LS', '488', '647', '425')
colours = np.repeat('black', len(metaData['Genotype']))                            
colours[np.array(metaData['Genotype']) == 'Kptn:Hom'] = 'red'  
f, axis = plt.subplots(5,1, figsize=(5,25))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(5):   
    axis[i].scatter(mixtureOutputList[0]['z-coordinate'], mixtureOutputList[i]['sigma_1'], color = colours)
    plt.xlabel('z-coordinate')
    plt.ylabel('estimated sd in intensity')
    axis[i].set_title(figureObjects[i] + '\n Estimated SD in Intensity Channel ' + channels[i])
plt.show()
f.savefig("figures/QC/ComponentSigmaVsZposition.png", bbox_inches='tight')  
plt.close(f)    # close the figure window
    
#Mean as function of batch:
figureObjects = ('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')
channels = ('568', '490LS', '488', '647', '425')
colours = np.repeat('black', len(metaData['Genotype']))                            
colours[np.array(metaData['Genotype']) == 'Kptn:Hom'] = 'red'  
f, axis = plt.subplots(5,1, figsize=(5,25))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(5):   
    axis[i].scatter(mixtureOutputList[0]['Cycle2-Batch'], mixtureOutputList[0]['mu_1'], color = colours)
    plt.xlabel('batch')
    plt.ylabel('estimated mean intensity')
    axis[i].set_title(figureObjects[i] + '\n Estimated Mean Intensity Channel ' + channels[i])
plt.show()
f.savefig("figures/QC/ComponentMeanVsBatch.png", bbox_inches='tight')  
plt.close(f)    # close the figure window

# Celltype assignement as function on z-position:

#Mean:
figureObjects = ('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')
channels = ('568', '490LS', '488', '647', '425')
colours = np.repeat('black', len(metaData['Genotype']))                            
colours[np.array(metaData['Genotype']) == 'Kptn:Hom'] = 'red'  
f, axis = plt.subplots(5,1, figsize=(5,25))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(5):   
    axis[i].scatter(cellTypeAssignments['z-coordinate'], cellTypeAssignments[figureObjects[i]], color = colours)
    plt.xlabel('batch')
    plt.ylabel('estimated abundance')
    axis[i].set_title(figureObjects[i] + '\n Abundance along z-coordinate ' + channels[i])
plt.show()
f.savefig("figures/QC/CelltypeAbundanceVsZcoordinate.png", bbox_inches='tight')  
plt.close(f)    # close the figure window
    
# Overview of for all channels and sections:

# Get all evaluation files:
root = '../data/KptnMouse/RNAscope'
pattern = "Objects_Population - Nuclei.txt"
allFiles = []
slideNames = []
measurementNames = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
            measurementNames.append(str.split(allFiles[-1], '/')[4])
            slideNames.append(str.split(measurementNames[-1],'__')[0])
            
numberOfSections = np.shape(mixtureOutput)[0]
f, axis = plt.subplots(2*numberOfSections,7, figsize=(21,2*3*numberOfSections))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
for i in range(numberOfSections):  
        
    print(i)    
    # Import data:
    index = np.where(np.array([allFiles[i].split('/')[4].split('__')[0] for i in range(len(allFiles))]) == np.array(mixtureOutputList[0]['SlideName'])[i])[0][0]
    # i.e. this is the index in the allFiles array that correponds to this measurment
    kptn_data_all = pd.read_csv(allFiles[index], sep = '\t' , skiprows = 8, header = 1)
    kptn_data = np.asarray(kptn_data_all[['Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Nucleus Alexa 568 Mean', 'Nuclei - Intensity Nucleus Atto 490LS Mean', 'Nuclei - Intensity Nucleus Alexa 488 Mean', 'Nuclei - Intensity Nucleus Alexa 647 Mean', 'Nuclei - Intensity Nucleus Atto 425 Mean']])

    # Filter out 1% smallest and 5% of largest nuclei as segmentation errors:
    volumes = np.asarray(kptn_data_all['Nuclei - Nucleus Volume [µm³]'])
    volumes = volumes[volumes.argsort()]
    minVol = volumes[int(np.round(len(volumes)*0.01))]
    maxVol = volumes[int(np.round(len(volumes)*0.95))]
    kptn_data = kptn_data[(kptn_data_all['Nuclei - Nucleus Volume [µm³]'] > minVol) & (kptn_data_all['Nuclei - Nucleus Volume [µm³]'] < maxVol),:]
    volumes = volumes[(volumes > minVol) & (volumes < maxVol)]
    kptn_data = kptn_data[kptn_data[:,1].argsort(),:]
    kptn_data_log = np.log2(kptn_data[:,2:])

    sectionNumber = np.zeros(np.shape(kptn_data_log)[0])
    count = 1
    sectionNumber[0] = count
    for j in range(1, np.shape(kptn_data_log)[0]):
        if abs(kptn_data[j,1] - kptn_data[j-1,1]) > 1000:
            if sum(sectionNumber == count) > 5000:
                count = count + 1
            else:
                sectionNumber[sectionNumber == count] = np.nan
        sectionNumber[j] = count

    # Overview of nuclei positions and celltype positions for all sections:
    
    file = open("data/" + measurementNames[int(np.where(np.array(slideNames) == np.array(mixtureOutputList[0]['SlideName'])[i])[0])] + 'Section' + str(np.array(mixtureOutputList[0]['Section'])[i])+ '_NucleixyPositions.pickle', 'rb')
    nuclei_positions = pickle.load(file)
    file.close()
    
    if os.path.exists("data/" + np.array(mixtureOutputList[channel]['SlideName'])[i] + 'Section' + str(np.array(mixtureOutputList[channel]['Section'])[i]) + '_FinalCelltypeAssignments.pickle'):
        file = open("data/" + np.array(mixtureOutputList[channel]['SlideName'])[i] + 'Section' + str(np.array(mixtureOutputList[channel]['Section'])[i]) + '_FinalCelltypeAssignments.pickle', 'rb')
        celltype_assignments = pickle.load(file)
        file.close()
        celltype_assignment_successful = True
    else:
        celltype_assignment_successful = False
        
    figureObjects = ('Nuclei', 'Unclassified', 'Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')
    objectsColours = ('black', 'grey', 'gold', 'pink','green', 'red', 'blue')
    channels = ('568', '490LS', '488', '647', '425')
    colourIndex = np.repeat(int(1), len(celltype_assignments))                            
    for j in range(5):
        colourIndex[celltype_assignments == (j+1)] = int(j+2)  
    
    axis[2*i,0].set_title(figureObjects[0] + ' Positions \n' + str(np.array(mixtureOutputList[0]['SlideName'])[i]) + ' Section' + str(np.array(mixtureOutputList[0]['Section'])[i]))
    axis[2*i,0].scatter(nuclei_positions['x_position'], nuclei_positions['y_position'], s = dotSize, color = objectsColours[0])
    
    if celltype_assignment_successful:

        for j in range(1,len(figureObjects)):

            axis[2*i,j].set_title(figureObjects[j] + ' Positions \n' + str(np.array(mixtureOutputList[0]['SlideName'])[i]) + ' Section' + str(np.array(mixtureOutputList[0]['Section'])[i]))
            axis[2*i,j].scatter(nuclei_positions['x_position'][celltype_assignments == (j-1)], nuclei_positions['y_position'][celltype_assignments == (j-1)], s = dotSize, color = objectsColours[j])

        for j in range(len(channels)):

            axis[2*i+1,j+2].set_title('Channel ' + channels[j] + ' Intensities and Classification \n' + str(np.array(mixtureOutputList[0]['SlideName'])[i]) + ' Section' + str(np.array(mixtureOutputList[0]['Section'])[i]))
            axis[2*i+1,j+2].scatter(kptn_data[sectionNumber == np.array(mixtureOutputList[0]['Section'])[i],0], kptn_data_log[sectionNumber == np.array(mixtureOutputList[0]['Section'])[i], j], s = dotSize, c = [objectsColours[colourIndex[i]] for i in range(len(colourIndex))]) 

plt.show()
f.savefig("figures/QC/AllSectionOverview.png", bbox_inches='tight')  
plt.close(f)    # close the figure window

# ### Try out spectral profile estimation:

# # Load both intensities and oligodendrocyte probabilities:

# # Get all evaluation files:
# root = '../data/KptnMouse/RNAscope'
# pattern = "Objects_Population - Nuclei.txt"
# allFiles = []
# slideNames = []
# measurementNames = []
# for path, subdirs, files in os.walk(root):
#     for name in files:
#         if fnmatch(name, pattern):
#             allFiles.append(os.path.join(path, name))
#             measurementNames.append(str.split(allFiles[-1], '/')[4])
#             slideNames.append(str.split(measurementNames[-1],'__')[0])

# for i in range(numberOfSections):  
        
#     print(i)    
#     # Import data:
#     index = np.where(np.array([allFiles[i].split('/')[4].split('__')[0] for i in range(len(allFiles))]) == mixtureOutputList[0]['SlideName'][i])[0][0]
#     # i.e. this is the index in the allFiles array that correponds to this measurment
#     kptn_data_all = pd.read_csv(allFiles[index], sep = '\t' , skiprows = 8, header = 1)
#     kptn_data = np.asarray(kptn_data_all[['Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Nucleus Alexa 568 Mean', 'Nuclei - Intensity Nucleus Atto 490LS Mean', 'Nuclei - Intensity Nucleus Alexa 488 Mean', 'Nuclei - Intensity Nucleus Alexa 647 Mean', 'Nuclei - Intensity Nucleus Atto 425 Mean']])

#     # Filter out 1% smallest and 5% of largest nuclei as segmentation errors:
#     volumes = np.asarray(kptn_data_all['Nuclei - Nucleus Volume [µm³]'])
#     volumes = volumes[volumes.argsort()]
#     minVol = volumes[int(np.round(len(volumes)*0.01))]
#     maxVol = volumes[int(np.round(len(volumes)*0.95))]
#     kptn_data = kptn_data[(kptn_data_all['Nuclei - Nucleus Volume [µm³]'] > minVol) & (kptn_data_all['Nuclei - Nucleus Volume [µm³]'] < maxVol),:]
#     volumes = volumes[(volumes > minVol) & (volumes < maxVol)]
#     kptn_data = kptn_data[kptn_data[:,1].argsort(),:]
#     kptn_data_log = np.log2(kptn_data[:,2:])

#     sectionNumber = np.zeros(np.shape(kptn_data_log)[0])
#     count = 1
#     sectionNumber[0] = count
#     for j in range(1, np.shape(kptn_data_log)[0]):
#         if abs(kptn_data[j,1] - kptn_data[j-1,1]) > 1000:
#             if sum(sectionNumber == count) > 5000:
#                 count = count + 1
#             else:
#                 sectionNumber[sectionNumber == count] = np.nan
#         sectionNumber[j] = count    

#     file = open("data/" + measurementNames[int(np.where(np.array(slideNames) == mixtureOutputList[0]['SlideName'][i])[0])] + 'Section' + str(mixtureOutputList[0]['Section'][i])+ '_NucleixyPositions.pickle', 'rb')
#     nuclei_positions = pickle.load(file)
#     file.close()
    
#     if os.path.exists("data/" + mixtureOutputList[channel]['SlideName'][i] + 'Section' + str(mixtureOutputList[channel]['Section'][i]) + '_FinalCelltypeAssignments.pickle'):
#         file = open("data/" + mixtureOutputList[channel]['SlideName'][i] + 'Section' + str(mixtureOutputList[channel]['Section'][i]) + '_FinalCelltypeAssignments.pickle', 'rb')
#         celltype_assignments = pickle.load(file)
#         file.close()
#         celltype_assignment_successful = True
#     else:
#         celltype_assignment_successful = False
    
#     if celltype_assignment_successful:
        
#         background = np.zeros(5)
#         for i in range(5):
#             background[i] = np.mean(np.sort(kptn_data[sectionNumber == 1,i+2])[:5000])
        
#         spectralProfile = np.zeros((5, 5))
#         for i in range(5):
#             spectralProfile[:,i] = np.mean((kptn_data[sectionNumber == 1, 2:][celltype_assignments == i+1] - background)/np.sum(kptn_data[sectionNumber == 1, 2:][celltype_assignments == i+1]-background, axis = 1)[:,None], axis = 0)
                
#         transformedData = np.matmul(kptn_data[:,2:],np.linalg.inv(spectralProfile))
#         index = 4
#         plt.hist(np.log2(transformedData[:,index]), bins = 100)
#         plt.hist(kptn_data_log[:,index], bins = 100)
        
#         ## Make a model to estimate spectral profile and background noise:



