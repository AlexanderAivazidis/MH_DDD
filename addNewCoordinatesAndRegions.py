import os
os.chdir('/home/jovyan/MH_DDD/')
from fnmatch import fnmatch
import pickle
import pandas as pd
import numpy as np

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

# Import metadata:
metaData = pd.read_excel('data/SectionReferenceSheet.xlsx')

# Now add channel intensities, region classification, cortical depth and radial distance into one large dataframe:
for i in range(len(evaluationFile)):
    print(i)
    # Import data:
    kptn_data_all = pd.read_csv(evaluationFile[i], sep = '\t' , skiprows = 8, header = 1)
    kptn_data = np.asarray(kptn_data_all[['Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Nucleus Alexa 568 Mean', 'Nuclei - Intensity Nucleus Atto 490LS Mean', 'Nuclei - Intensity Nucleus Alexa 488 Mean', 'Nuclei - Intensity Nucleus Alexa 647 Mean', 'Nuclei - Intensity Nucleus Atto 425 Mean']])
    
    # Filter out 1% smallest and 5% of largest nuclei as segmentation errors:
    volumes = np.asarray(kptn_data_all['Nuclei - Nucleus Volume [µm³]'])
    volumes = volumes[volumes.argsort()]
    minVol = volumes[int(np.round(len(volumes)*0.01))]
    maxVol = volumes[int(np.round(len(volumes)*0.95))]
    kptn_data = kptn_data[(kptn_data_all['Nuclei - Nucleus Volume [µm³]'] > minVol) & (kptn_data_all['Nuclei - Nucleus Volume [µm³]'] < maxVol),:]
    volumes = volumes[(volumes > minVol) & (volumes < maxVol)]
    kptn_data = kptn_data[np.flipud(kptn_data[:,1].argsort()),:]
    sectionNumber = np.zeros(np.shape(kptn_data)[0])
    count = 1
    sectionNumber[0] = count
    for j in range(1, np.shape(kptn_data)[0]):
        if abs(kptn_data[j,1] - kptn_data[j-1,1]) > 1000:
            if sum(sectionNumber == count) > 5000:
                count = count + 1
            else:
                sectionNumber[sectionNumber == count] = np.nan
        sectionNumber[j] = count
    uniqueSectionNumbersWithNaN = np.unique(sectionNumber)    
    uniqueSectionNumbers = uniqueSectionNumbersWithNaN[~np.isnan(uniqueSectionNumbersWithNaN)]
    
    # Import corresponding data on region and new coordinates:
    
    #Regions:
    file = open("data/nucleiPositions/" + measurmentNames[i] + "Section" + sectionID[i] + "_NucleiRegions.pickle", 'rb')
    regions = pickle.load(file)
    file.close()
    #Cortical Depth:
    file = open("data/nucleiPositions/" + measurmentNames[i] + "Section" + sectionID[i] + "_NucleiNormalizedCorticalDepth.pickle", 'rb')
    corticalDepth = pickle.load(file)
    file.close()
    #Radial Distance:
    file = open("data/nucleiPositions/" + measurmentNames[i] + "Section" + sectionID[i] + "_NucleiRadialDistance.pickle", 'rb')
    radialDistance = pickle.load(file)
    file.close()
    #MedialCortex:
    file = open("data/nucleiPositions/" + measurmentNames[i] + "Section" + sectionID[i] + "_NucleiMedialCortex.pickle", 'rb')
    medialCortex = pickle.load(file)
    file.close()
    #Upper Layers:
    file = open("data/nucleiPositions/" + measurmentNames[i] + "Section" + sectionID[i] + "_NucleiUpperCorticalLayers.pickle", 'rb')
    upperLayers = pickle.load(file)
    file.close()
    #Bubbles:
    file = open("data/nucleiPositions/" + measurmentNames[i] + "Section" + sectionID[i] + "_NucleiBubbles.pickle", 'rb')
    bubbles = pickle.load(file)
    file.close()
    
    # Save all data on nuclei in cortex in one dataframe:
    kptn_data = kptn_data[sectionNumber == int(sectionID[i]),]
    kptn_data = kptn_data
    relevantMetadata = metaData[(metaData['Automatic SlideID - Cycle 2'] == slideNames[i]) & (metaData['Section Number'] == int(sectionID[i]))]
    
    columnNames = ('SlideName', 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'Genotype', 'MouseID', 'x-coordinate', 'y-coordinate', 'x_dash-coordinate', 'y_dash_coordinate', 'Bubble', 'Region', 'Hemisphere', 'Upper-Lower', 'Medial-Lateral', '568 Intensity', '490LS Intensity', '488 Intensity', '647 Intensity', '425 Intensity')
    cortexData = pd.DataFrame(index=range(np.shape(kptn_data)[0]), columns=columnNames)
    cortexData['SlideName'] = slideNames[i]
    cortexData['Section'] = sectionID[i]
    cortexData['Cycle1-Batch'] = int(relevantMetadata['Batch - Cycle 1'])
    cortexData['Cycle2-Batch'] = int(relevantMetadata['Batch -- Cycle 2'])
    cortexData['z-coordinate'] = int(relevantMetadata['Figure Number - By Eye'])
    cortexData['Genotype'] = np.array(relevantMetadata['Genotype'])[0]
    cortexData['MouseID'] = np.array(relevantMetadata['MouseID'])[0]
    cortexData['x-coordinate'] = kptn_data[:,0]
    cortexData['y-coordinate'] = kptn_data[:,1]
    cortexData['x_dash-coordinate'] = radialDistance #[r for r in radialDistance if r is not None]
    cortexData['y_dash-coordinate'] = corticalDepth #[c for c in corticalDepth if c is not None]
    cortexData['Region'] = 'Cortex'
    cortexData['Bubbles'] = bubbles #[bubbles[j] for j in range(len(regions)) if regions[j] != 0]
    cortexData['Hemisphere'] = regions #[regions[j] for j in range(len(regions)) if regions[j] != 0]
    cortexData['Upper-Lower'] = upperLayers #[upperLayers[j] for j in range(len(upperLayers)) if upperLayers[j] is not None]
    cortexData['Medial-Lateral'] = medialCortex #[medialCortex[j] for j in range(len(medialCortex)) if medialCortex[j] is not None]
    cortexData['568 Intensity'] = kptn_data[:,2]
    cortexData['490LS Intensity'] = kptn_data[:,3]
    cortexData['488 Intensity'] = kptn_data[:,4]
    cortexData['647 Intensity'] = kptn_data[:,5]
    cortexData['425 Intensity'] = kptn_data[:,6]
    
    pickle_out = open("data/cortexData/" + slideNames[i] + 'Section' + sectionID[i] + 'cortexData_.pickle',"wb")
    pickle.dump(cortexData, pickle_out)
    pickle_out.close()


    
    

            
            
            