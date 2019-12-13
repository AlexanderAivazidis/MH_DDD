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

# Overview of nuclei numbers as function of z coordinates:
metaData = pd.read_excel('data/SectionReferenceSheet.xlsx')
path = 'data/nucleiPositions'
allFiles = [f for f in listdir(path) if isfile(join(path, f))]
slideNames = [allFiles[i].split('__')[0] for i in range(len(allFiles))]
measurementNames = [allFiles[i].split('Section')[0] for i in range(len(allFiles))]

# Overview of nuclei numbers as function of z coordinate in cortex:
columnNames = ('SlideName', 'AlternativeSlideName' 'Section', 'Cycle1-Batch', 'Cycle2-Batch', 'z-coordinate', 'numberOfNuclei')
nucleiNumbers = pd.DataFrame(index=range(np.shape(metaData)[0]), columns=columnNames)
nucleiNumbers['SlideName'] = metaData['Automatic SlideID - Cycle 2']
nucleiNumbers['AlternativeSlideName'] = metaData['SlideID - Cycle 2']
nucleiNumbers['Section'] = metaData['Section Number']
nucleiNumbers['Cycle1-Batch'] = metaData['Batch - Cycle 1']
nucleiNumbers['Cycle2-Batch'] = metaData['Batch -- Cycle 2']
nucleiNumbers['z-coordinate'] = metaData['Figure Number - Sectioning']

for i in range(np.shape(nucleiNumbers)[0]):
    file = open("data/nucleiPositions/" + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleixyPositions.pickle', 'rb')
    nuclei_positions = pickle.load(file)
    file.close()
    nucleiNumbers['numberOfNuclei'][i] = len(nuclei_positions['x_position'])

colours = np.repeat('black', len(metaData['Genotype']))                            
colours[np.array(metaData['Genotype']) == 'Kptn:Hom'] = 'red'
f, axis = plt.subplots(1,1, figsize=(8,5))
plt.scatter(nucleiNumbers['z-coordinate'], nucleiNumbers['numberOfNuclei'], color = colours)
plt.xlabel('z-coordinate')
plt.ylabel('total number of nuclei')
f.savefig("figures/NucsleiNumbersVsZposition.png", bbox_inches='tight')  
plt.close(f)    # close the figure window

# Classify nuclei into regions: Left hemisphere, right hemisphere, bubbles

numberOfSections = 0
for i in range(np.shape(nucleiNumbers)[0]):
    if os.path.exists("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml'):
        numberOfSections += 1

f, axis = plt.subplots(numberOfSections,1, figsize=(7,numberOfSections*7))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
count = -1
for i in range(np.shape(nucleiNumbers)[0]):
    if os.path.exists("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml'):
        file = open("data/nucleiPositions/" + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleixyPositions.pickle', 'rb')
        nuclei_positions = pickle.load(file)
        file.close()
        count += 1
        print(i)
        with open("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml', 'r') as stream:
                segmentationMasks = yaml.safe_load(stream)
        offset = pd.read_csv("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_offset.txt', header = None)
        offset = np.array(offset)[0]
        sectionNumber = nucleiNumbers['Section'][i]
        cortex = np.zeros(len(nuclei_positions['x_position']))
        isbubble = np.zeros(len(nuclei_positions['x_position']))
        # Left Hemisphere:
        positions =  nested_lookup('position', segmentationMasks)
        wm_masks = nested_lookup('wm', segmentationMasks)
        pia_masks = nested_lookup('pia', segmentationMasks)
        firstSplit = str.split(positions[(2*sectionNumber)-2][0], ';')
        polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))
        for j in range(len(cortex)):
            point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000+offset[1])
            if polygon.contains(point):
                cortex[j] = 1  
        # Right Hemisphere:
        firstSplit = str.split(positions[(2*sectionNumber)-1][0], ';')
        polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))
        for j in range(len(cortex)):
            point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
            if polygon.contains(point):
                cortex[j] = 2  
        colours = np.repeat('black', len(nuclei_positions['x_position']))
        colours[cortex == 1] = 'green'
        colours[cortex == 2] = 'green'
        # Also mark white matter surface and bubbles in figures:
        if len(wm_masks[int((2/(len(positions)/len(wm_masks))*sectionNumber)-1)]) > 0:
            wml = str.split(wm_masks[(2*sectionNumber)-2][0], ';')
            wml = LineString(np.asarray([str.split(wml[i], ' ') for i in range(len(wml))]).astype(np.float))
            x_wml, y_wml = wml.xy
            x_wml = [(x_wml[i]-offset[0])*1000 for i in range(len(x_wml))]
            y_wml = [(y_wml[i]-offset[1])*1000 for i in range(len(y_wml))]
            pial = str.split(pia_masks[(2*sectionNumber)-2][0], ';')
            pial = LineString(np.asarray([str.split(pial[i], ' ') for i in range(len(pial))]).astype(np.float)) 
            x_pial, y_pial = pial.xy
            x_pial = [(x_pial[i]-offset[0])*1000 for i in range(len(x_pial))]
            y_pial = [(y_pial[i]-offset[1])*1000 for i in range(len(y_pial))]
            wmr = str.split(wm_masks[(2*sectionNumber)-1][0], ';')
            wmr = LineString(np.asarray([str.split(wmr[i], ' ') for i in range(len(wmr))]).astype(np.float))
            x_wmr, y_wmr = wmr.xy
            x_wmr = [(x_wmr[i]-offset[0])*1000 for i in range(len(x_wmr))]
            y_wmr = [(y_wmr[i]-offset[1])*1000 for i in range(len(y_wmr))]
            piar = str.split(pia_masks[(2*sectionNumber)-1][0], ';')
            piar = LineString(np.asarray([str.split(piar[i], ' ') for i in range(len(piar))]).astype(np.float)) 
            x_piar, y_piar = piar.xy
            x_piar = [(x_piar[i]-offset[0])*1000 for i in range(len(x_piar))]
            y_piar = [(y_piar[i]-offset[1])*1000 for i in range(len(y_piar))]
            
        if os.path.exists("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_BB.yml'):
            with open("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_BB.yml', 'r') as stream:
                bubbleMasks = yaml.safe_load(stream)
            bubbles = nested_lookup('position', bubbleMasks)
            for j in range(len(bubbles)):
                bubble = str.split(bubbles[j][0], ';')
                bubble = Polygon(np.asarray([str.split(bubble[i], ' ') for i in range(len(bubble))]).astype(np.float))
                for l in range(len(cortex)):
                    point = Point(nuclei_positions['x_position'][l]/1000+offset[0], nuclei_positions['y_position'][l]/1000 + offset[1])
                    if bubble.contains(point):
                        colours[l] = 'red'
                        isbubble[l] = 1
                
#         axis[count].scatter(nuclei_positions['x_position'], nuclei_positions['y_position'], c = colours, s = 0.05)
#         if len(segmentationMasks['slices'][(2*sectionNumber)-2]['pia']) > 0:
#             axis[count].plot(x_pial, y_pial, color = 'blue')
#         if len(segmentationMasks['slices'][(2*sectionNumber)-2]['wm']) > 0:
#             axis[count].plot(x_wml, y_wml, color = 'orange')
#         if len(segmentationMasks['slices'][(2*sectionNumber)-1]['pia']) > 0:    
#             axis[count].plot(x_piar, y_piar, color = 'blue')
#         if len(segmentationMasks['slices'][(2*sectionNumber)-1]['wm']) > 0:
#             axis[count].plot(x_wmr, y_wmr, color = 'orange')
#         axis[count].set_title('Left (red) and Right (green) Cortex Segmentation \n' + str(np.array(nucleiNumbers['SlideName'])[i]) + ' Section' + str(np.array(nucleiNumbers['Section'])[i]))

        pickle_out = open("data/nucleiPositions/"  + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleiRegions.pickle',"wb")
        pickle.dump(cortex, pickle_out)
        pickle_out.close()
        pickle_out = open("data/nucleiPositions/"  + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleiBubbles.pickle',"wb")
        pickle.dump(isbubble, pickle_out)
        pickle_out.close()
        
f.savefig("figures/SegmentationMasksCortex.png", bbox_inches='tight')  
plt.close(f)    # close the figure window

# Now add new coordinates to the nuclei: normalized cortical depth, radial distance and use these to define upper and lower cortical layers + medial and lateral parts of the cortex

numberOfSections = 0
for i in range(np.shape(nucleiNumbers)[0]):
    if os.path.exists("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml'):
        with open("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml', 'r') as stream:
            segmentationMasks = yaml.safe_load(stream)  
        sectionNumber = nucleiNumbers['Section'][i]
        if len(segmentationMasks['slices'][(2*sectionNumber)-2]['wm']) > 0:
            numberOfSections += 1

f, axis = plt.subplots(numberOfSections, 4, figsize=(7*4,numberOfSections*7))
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.facecolor'] = 'white'
dotSize = 0.5
count = -1
for i in range(np.shape(nucleiNumbers)[0]):
    if os.path.exists("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml'):
        file = open("data/nucleiPositions/" + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleixyPositions.pickle', 'rb')
        nuclei_positions = pickle.load(file)
        file.close()
        print(i)
        with open("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml', 'r') as stream:
                segmentationMasks = yaml.safe_load(stream)
        offset = pd.read_csv("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_offset.txt', header = None)
        offset = np.array(offset)[0]
        sectionNumber = nucleiNumbers['Section'][i]
        cortex = np.zeros(len(nuclei_positions['x_position']))
        # Left Hemisphere:
        firstSplit = str.split(segmentationMasks['slices'][(2*sectionNumber)-2]['rois']['position'][0], ';')
        polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))
        for j in range(len(cortex)):
            point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000+offset[1])
            if polygon.contains(point):
                cortex[j] = 1  
        # Right Hemisphere:
        firstSplit = str.split(segmentationMasks['slices'][(2*sectionNumber)-1]['rois']['position'][0], ';')
        polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))
        for j in range(len(cortex)):
            point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
            if polygon.contains(point):
                cortex[j] = 2  
        if len(segmentationMasks['slices'][(2*sectionNumber)-2]['wm']) > 0:
            count += 1
            wml = str.split(segmentationMasks['slices'][(2*sectionNumber)-2]['wm'][0], ';')
            wml = LineString(np.asarray([str.split(wml[i], ' ') for i in range(len(wml))]).astype(np.float))
            x_wml, y_wml = wml.xy
            x_wml = [(x_wml[i]-offset[0])*1000 for i in range(len(x_wml))]
            y_wml = [(y_wml[i]-offset[1])*1000 for i in range(len(y_wml))]
        
            wmr = str.split(segmentationMasks['slices'][(2*sectionNumber)-1]['wm'][0], ';')
            wmr = LineString(np.asarray([str.split(wmr[i], ' ') for i in range(len(wmr))]).astype(np.float))
            x_wmr, y_wmr = wmr.xy
            x_wmr = [(x_wmr[i]-offset[0])*1000 for i in range(len(x_wmr))]
            y_wmr = [(y_wmr[i]-offset[1])*1000 for i in range(len(y_wmr))]
            
            pial = str.split(segmentationMasks['slices'][(2*sectionNumber)-2]['pia'][0], ';')
            pial = LineString(np.asarray([str.split(pial[i], ' ') for i in range(len(pial))]).astype(np.float)) 
            x_pial, y_pial = pial.xy
            x_pial = [(x_pial[i]-offset[0])*1000 for i in range(len(x_pial))]
            y_pial = [(y_pial[i]-offset[1])*1000 for i in range(len(y_pial))]
        
            wmr = str.split(segmentationMasks['slices'][(2*sectionNumber)-1]['wm'][0], ';')
            wmr = LineString(np.asarray([str.split(wmr[i], ' ') for i in range(len(wmr))]).astype(np.float))
            x_wmr, y_wmr = wmr.xy
            x_wmr = [(x_wmr[i]-offset[0])*1000 for i in range(len(x_wmr))]
            y_wmr = [(y_wmr[i]-offset[1])*1000 for i in range(len(y_wmr))]
 
            piar = str.split(segmentationMasks['slices'][(2*sectionNumber)-1]['pia'][0], ';')
            piar = LineString(np.asarray([str.split(piar[i], ' ') for i in range(len(piar))]).astype(np.float)) 
            x_piar, y_piar = piar.xy
            x_piar = [(x_piar[i]-offset[0])*1000 for i in range(len(x_piar))]
            y_piar = [(y_piar[i]-offset[1])*1000 for i in range(len(y_piar))]

            # Compute new coordinates for left hemisphere:

            radialDistanceL = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 1)):
                point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
                radialDistanceL[j] = 1-wml.project(point, normalized = True)
            wmDistanceL = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 1)):
                point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
                wmDistanceL[j] = wml.distance(point)
            pialDistanceL = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 1)):
                point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
                pialDistanceL[j] = pial.distance(point)
            norm_corticalDepthL = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 1)):
                norm_corticalDepthL[j] = wmDistanceL[j]/(wmDistanceL[j] + pialDistanceL[j])

            # Compute new coordinates for right hemisphere:

            radialDistanceR = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 2)):
                point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
                radialDistanceR[j] = wmr.project(point, normalized = True)
            wmDistanceR = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 2)):
                point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
                wmDistanceR[j] = wmr.distance(point)
            pialDistanceR = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 2)):
                point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000 + offset[1])
                pialDistanceR[j] = piar.distance(point)
            norm_corticalDepthR = np.repeat(None, len(nuclei_positions['x_position']))
            for j in np.nditer(np.where(cortex == 2)):
                norm_corticalDepthR[j] = wmDistanceR[j]/(wmDistanceR[j] + pialDistanceR[j])
                
            # We want the radialDistanceR to increase with x coordinate:
            if np.corrcoef(nuclei_positions['x_position'][cortex == 2], radialDistanceR[cortex == 2].astype(float))[1,0] < 0:
                radialDistanceR[cortex == 2] = 1-radialDistanceR[cortex == 2]
            # We want the radialDistanceL to decrease with x coordinate:
            if np.corrcoef(nuclei_positions['x_position'][cortex == 1], radialDistanceL[cortex == 1].astype(float))[1,0] > 0:
                radialDistanceL[cortex == 1] = 1-radialDistanceL[cortex == 1]
            
            # Now define some new regions within the cortex based on these coordinates:
            upperLayersR = np.repeat(None, len(norm_corticalDepthR))
            upperLayersR[cortex == 2] = [0 if n > 0.5 else 1 for n in norm_corticalDepthR[cortex == 2]]
            upperLayersL = np.repeat(None, len(norm_corticalDepthL))
            upperLayersL[cortex == 1] = [0 if n > 0.5 else 1 for n in norm_corticalDepthL[cortex == 1]]
            
            medialPartR = np.repeat(None, len(radialDistanceR))
            medialPartR[cortex == 2] = [0 if r < 0.5 else 1 for r in radialDistanceR[cortex == 2]]
            medialPartL = np.repeat(None, len(radialDistanceL))
            medialPartL[cortex == 1] = [0 if r < 0.5 else 1 for r in radialDistanceL[cortex == 1]]
            
            cmap = sns.cubehelix_palette(as_cmap=True)

            axis[count,1].scatter(nuclei_positions['x_position'][cortex == 0], nuclei_positions['y_position'][cortex == 0], c = 'black', s = 0.05)
            points = axis[count,1].scatter(nuclei_positions['x_position'][cortex ==1], nuclei_positions['y_position'][cortex ==1], c=radialDistanceL[cortex == 1], s=0.05, cmap=cmap)
            points = axis[count,1].scatter(nuclei_positions['x_position'][cortex ==2], nuclei_positions['y_position'][cortex ==2], c=radialDistanceR[cortex == 2], s=0.05, cmap=cmap)
            axis[count,1].set_title('Normalized Radial Distance \n Slide ' + str(np.array(nucleiNumbers['SlideName'])[i]) + ' Section' + str(np.array(nucleiNumbers['Section'])[i]))

            axis[count,0].scatter(nuclei_positions['x_position'][cortex == 0], nuclei_positions['y_position'][cortex == 0], c = 'black', s = 0.05)
            points = axis[count,0].scatter(nuclei_positions['x_position'][cortex ==1], nuclei_positions['y_position'][cortex ==1], c=norm_corticalDepthL[cortex == 1], s=0.05, cmap=cmap)
            points = axis[count,0].scatter(nuclei_positions['x_position'][cortex ==2], nuclei_positions['y_position'][cortex ==2], c=norm_corticalDepthR[cortex == 2], s=0.05, cmap=cmap)
            axis[count,0].set_title('Normalized CorticalDepth \n Slide ' + str(np.array(nucleiNumbers['SlideName'])[i]) + ' Section' + str(np.array(nucleiNumbers['Section'])[i]))
            
            axis[count,2].scatter(nuclei_positions['x_position'][cortex == 0], nuclei_positions['y_position'][cortex == 0], c = 'black', s = 0.05)
            points = axis[count,2].scatter(nuclei_positions['x_position'][cortex ==1], nuclei_positions['y_position'][cortex ==1], c=upperLayersL[cortex == 1], s=0.05, cmap=cmap)
            points = axis[count,2].scatter(nuclei_positions['x_position'][cortex ==2], nuclei_positions['y_position'][cortex ==2], c=upperLayersR[cortex == 2], s=0.05, cmap=cmap)
            axis[count,2].set_title('Upper and Lower Cortical Layers \n Slide ' + str(np.array(nucleiNumbers['SlideName'])[i]) + ' Section' + str(np.array(nucleiNumbers['Section'])[i]))
            
            axis[count,3].scatter(nuclei_positions['x_position'][cortex == 0], nuclei_positions['y_position'][cortex == 0], c = 'black', s = 0.05)
            points = axis[count,3].scatter(nuclei_positions['x_position'][cortex ==1], nuclei_positions['y_position'][cortex ==1], c=medialPartL[cortex == 1], s=0.05, cmap=cmap)
            points = axis[count,3].scatter(nuclei_positions['x_position'][cortex ==2], nuclei_positions['y_position'][cortex ==2], c=medialPartR[cortex == 2], s=0.05, cmap=cmap)
            axis[count,3].set_title('Medial and Lateral Parts of Cortex \n Slide ' + str(np.array(nucleiNumbers['SlideName'])[i]) + ' Section' + str(np.array(nucleiNumbers['Section'])[i]))
            
            norm_corticalDepth = norm_corticalDepthL
            norm_corticalDepth[cortex == 2] = norm_corticalDepthR[cortex == 2]
            radialDistance = radialDistanceL
            radialDistance[cortex == 2] = radialDistanceR[cortex == 2]
            upperLayers = upperLayersL
            upperLayers[cortex == 2] = upperLayersR[cortex == 2]
            medialPart = medialPartL
            medialPart[cortex == 2] = medialPartR[cortex == 2]
            
            pickle_out = open("data/nucleiPositions/"  + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleiNormalizedCorticalDepth.pickle',"wb")
            pickle.dump(norm_corticalDepth, pickle_out)
            pickle_out.close()

            pickle_out = open("data/nucleiPositions/"  + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleiRadialDistance.pickle',"wb")
            pickle.dump(radialDistance, pickle_out)
            pickle_out.close()
            
            pickle_out = open("data/nucleiPositions/"  + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleiUpperCorticalLayers.pickle',"wb")
            pickle.dump(upperLayers, pickle_out)
            pickle_out.close()

            pickle_out = open("data/nucleiPositions/"  + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleiMedialCortex.pickle',"wb")
            pickle.dump(medialPart, pickle_out)
            pickle_out.close()
                
f.colorbar(points)            

f.savefig("figures/NaturalCoordinatesCortex.png", bbox_inches='tight')  
plt.close(f)    # close the figure window

# Make a plot of total number of nuclei in cortex for captain and wt as function of z:
totalNumber = np.zeros(numberOfSections)
colour = np.repeat('black', numberOfSections)
count = -1
zPositions = np.zeros(numberOfSections)

for i in range(np.shape(nucleiNumbers)[0]):
    if os.path.exists("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml'):
        file = open("data/nucleiPositions/" + np.unique(measurementNames)[int(np.where(np.array(np.unique(slideNames)) == np.array(nucleiNumbers['SlideName'])[i])[0])] + 'Section' + str(np.array(nucleiNumbers['Section'])[i])+ '_NucleixyPositions.pickle', 'rb')
        nuclei_positions = pickle.load(file)
        file.close()
        count += 1
        print(i)
        with open("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_CTX.yml', 'r') as stream:
                segmentationMasks = yaml.safe_load(stream)
        offset = pd.read_csv("data/segmentationMasks/" + nucleiNumbers['AlternativeSlideName'][i] + '_offset.txt', header = None)
        offset = np.array(offset)[0]
        sectionNumber = nucleiNumbers['Section'][i]
        cortex = np.zeros(len(nuclei_positions['x_position']))
        # Left Hemisphere:
        firstSplit = str.split(segmentationMasks['slices'][(2*sectionNumber)-2]['rois']['position'][0], ';')
        polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))
        for j in range(len(cortex)):
            point = Point(nuclei_positions['x_position'][j]/1000+offset[0], nuclei_positions['y_position'][j]/1000+offset[1])
            if polygon.contains(point):
                cortex[j] = 1  
        # Right Hemisphere:
        firstSplit = str.split(segmentationMasks['slices'][(2*sectionNumber)-1]['rois']['position'][0], ';')
        polygon = Polygon(np.asarray([str.split(firstSplit[i], ' ') for i in range(len(firstSplit))]).astype(np.float))
        for j in range(len(cortex)):
            if polygon.contains(point):
                cortex[j] = 2  
        if metaData['Genotype'][i] == 'Kptn:Hom':
            colour[count] = 'red'
        totalNumber[count] = sum(cortex == 1) + sum(cortex == 2)
        zPositions[count] = np.array(metaData['Figure Number - Sectioning'])[i]