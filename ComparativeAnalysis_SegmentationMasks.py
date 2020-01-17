### Compare the segmentation masks to see possible differences in size between Kptn and WT ###

import os
os.chdir('/home/jovyan/MH_DDD/')
from fnmatch import fnmatch
import pickle
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import yaml
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString
from matplotlib.pyplot import figure
from nested_lookup import nested_lookup
import seaborn as sns

# Import all segmentation masks that have pial/wm surface segmented:
metaData = pd.read_excel('data/SectionReferenceSheet.xlsx')

numberOfSections = 0
for i in range(np.shape(metaData)[0]):
    if os.path.exists("data/segmentationMasks/" + metaData['SlideID - Cycle 2'][i] + '_CTX.yml'):
        with open("data/segmentationMasks/" + metaData['SlideID - Cycle 2'][i] + '_CTX.yml', 'r') as stream:
            segmentationMasks = yaml.safe_load(stream)  
        sectionNumber = metaData['Section Number'][i]
        if len(nested_lookup('wm', segmentationMasks)) > 0:
            numberOfSections += 1

columnNames = ('Index', 'LeftCortex', 'RightCortex', 'Genotype', 'MouseID', 'z-coordinate')
areaData = pd.DataFrame(index=range(numberOfSections), columns=columnNames)
count = -1
for i in range(np.shape(metaData)[0]):
    if os.path.exists("data/segmentationMasks/" + metaData['SlideID - Cycle 2'][i] + '_CTX.yml'):
        with open("data/segmentationMasks/" + metaData['SlideID - Cycle 2'][i] + '_CTX.yml', 'r') as stream:
            segmentationMasks = yaml.safe_load(stream)  
        sectionNumber = metaData['Section Number'][i]
        if len(nested_lookup('wm', segmentationMasks)) > 0:
            count += 1
            positions =  nested_lookup('position', segmentationMasks)
            firstSplit = str.split(positions[(2*sectionNumber)-2][0], ';')
            areaData['LeftCortex'][count] = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float)).area
            firstSplit = str.split(positions[(2*sectionNumber)-1][0], ';')
            areaData['RightCortex'][count] = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float)).area
            areaData['Index'][count] = i
            areaData['Genotype'][count] = metaData['Genotype'][i]
            areaData['MouseID'][count] = metaData['MouseID'][i]
            areaData['z-coordinate'][count] = metaData['Figure Number - By Eye'][i]
            
# Plot size of Kptn and WT cortex segmentation masks area as function of z coordinate:

colours = np.repeat('black', np.shape(areaData)[0])
colours[areaData['Genotype'] == 'Kptn:Hom'] = 'red'

f, axis = plt.subplots(1, 1, figsize=(15,7))
axis.scatter(areaData['z-coordinate'], areaData['LeftCortex'], c = colours)
axis.scatter(areaData['z-coordinate'], areaData['RightCortex'], c = colours)
