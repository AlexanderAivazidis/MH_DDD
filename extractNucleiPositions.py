import os
os.chdir('/home/jovyan/MH_DDD/')
from fnmatch import fnmatch
import pickle
import pandas as pd
import numpy as np

## Extract and save nuclei positions:

# Get all evaluation files:
root = '../data/KptnMouse/RNAscope'
pattern = "Objects_Population - Nuclei.txt"
allFiles = []
slideNames = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
            slideNames.append(str.split(allFiles[-1], '/')[4])
    
for slide in range(len(allFiles)):
    
    # Import data:
    kptn_data_all = pd.read_csv(allFiles[slide], sep = '\t' , skiprows = 8, header = 1)
    kptn_data = np.asarray(kptn_data_all[['Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Nucleus Alexa 568 Mean', 'Nuclei - Intensity Nucleus Atto 490LS Mean', 'Nuclei - Intensity Nucleus Alexa 488 Mean', 'Nuclei - Intensity Nucleus Alexa 647 Mean', 'Nuclei - Intensity Nucleus Atto 425 Mean']])
    
    # Filter out 1% smallest and 5% of largest nuclei as segmentation errors:

    volumes = np.asarray(kptn_data_all['Nuclei - Nucleus Volume [µm³]'])
    volumes = volumes[volumes.argsort()]
    minVol = volumes[int(np.round(len(volumes)*0.01))]
    maxVol = volumes[int(np.round(len(volumes)*0.95))]

    kptn_data = kptn_data[(kptn_data_all['Nuclei - Nucleus Volume [µm³]'] > minVol) & (kptn_data_all['Nuclei - Nucleus Volume [µm³]'] < maxVol),:]
    volumes = volumes[(volumes > minVol) & (volumes < maxVol)]
    kptn_data = kptn_data[np.flipud(kptn_data[:,1].argsort()),:]
    kptn_data_log = np.log2(kptn_data[:,2:])

    sectionNumber = np.zeros(np.shape(kptn_data_log)[0])
    count = 1
    sectionNumber[0] = count
    for i in range(1, np.shape(kptn_data_log)[0]):
        if abs(kptn_data[i,1] - kptn_data[i-1,1]) > 1000:
            if sum(sectionNumber == count) > 5000:
                count = count + 1
            else:
                sectionNumber[sectionNumber == count] = np.nan
        sectionNumber[i] = count

    uniqueSectionNumbersWithNaN = np.unique(sectionNumber)    
    uniqueSectionNumbers = uniqueSectionNumbersWithNaN[~np.isnan(uniqueSectionNumbersWithNaN)]
        
    for section in np.unique(uniqueSectionNumbers):
        # Save location of remaining nuclei:
        nucleiPositions = {"x_position": kptn_data[sectionNumber == section,0], "y_position": kptn_data[sectionNumber == section,1]}
        pickle_out = open("data/nucleiPositions/" + slideNames[slide] + 'Section' + str(int(section)) + '_NucleixyPositions.pickle',"wb")
        pickle.dump(nucleiPositions, pickle_out)
        pickle_out.close()

            