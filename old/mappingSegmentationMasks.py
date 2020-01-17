from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
os.chdir('/home/jovyan/MH_DDD/')
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from matplotlib.pyplot import figure

# Load nuclei coordinates, segmentation masks and offset:

import yaml

with open("data/segmentationMasks/MH_DDD_007B_CTX_masks.yml", 'r') as stream:
        segmentationMasks = yaml.safe_load(stream)

offset = pd.read_csv('data/segmentationMasks/MH_DDD_007B_offset.txt', header = None)
offset = np.array(offset)[0]

file = open('data/190816_040549-V__2019-08-16T04_35_11-Measurement 1bSection1_NucleixyPositions.pickle', 'rb')
nuclei_positions = pickle.load(file)
file.close()

firstSplit = str.split(segmentationMasks['slices'][5]['rois']['position'][0], ';')
polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))

cortex = np.zeros(len(nuclei_positions['x_position']))
for i in range(len(cortex)):
    point = Point((nuclei_positions['x_position'][i]-offset[0])/1000, (nuclei_positions['y_position'][i]-offset[1])/1000)
    if polygon.contains(point):
        cortex[i] = 1
        
firstSplit = str.split(segmentationMasks['slices'][4]['rois']['position'][0], ';')
polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))

for i in range(len(cortex)):
    point = Point((nuclei_positions['x_position'][i]-offset[0])/1000, (nuclei_positions['y_position'][i]-offset[1])/1000)
    if polygon.contains(point):
        cortex[i] = 2

colours = np.repeat('black', len(nuclei_positions['x_position']))
colours[cortex == 1] = 'red'
colours[cortex == 2] = 'green'
f = figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(nuclei_positions['x_position'], nuclei_positions['y_position'], c = colours, s = 0.05)

plt.show()
f.savefig("figures/QC/SegmentationMaskAttempt.png", bbox_inches='tight')  
plt.close(f)    # close the figure window