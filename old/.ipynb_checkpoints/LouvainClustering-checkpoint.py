import numpy as np
import pandas as pd
import scanpy as sc
import os
os.chdir('/home/jovyan/MH_DDD/')
from fnmatch import fnmatch

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

slide = 0
# Import data:
kptn_data_all = sc.read_csv(allFiles[slide], sep = '\t' , skiprows = 8, header = 1)
kptn_data = np.asarray(kptn_data_all[['Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Nucleus Alexa 568 Mean', 'Nuclei - Intensity Nucleus Atto 490LS Mean', 'Nuclei - Intensity Nucleus Alexa 488 Mean', 'Nuclei - Intensity Nucleus Alexa 647 Mean', 'Nuclei - Intensity Nucleus Atto 425 Mean']])
channelOrder = ('568', '490LS', '488', '647', '425')
celltypeOrder = ('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron')

# Filter out 1% smallest and 5% of largest nuclei as segmentation errors:

volumes = np.asarray(kptn_data_all['Nuclei - Nucleus Volume [µm³]'])
volumes = volumes[volumes.argsort()]
minVol = volumes[int(np.round(len(volumes)*0.01))]
maxVol = volumes[int(np.round(len(volumes)*0.95))]

kptn_data = kptn_data[(kptn_data_all['Nuclei - Nucleus Volume [µm³]'] > minVol) & (kptn_data_all['Nuclei - Nucleus Volume [µm³]'] < maxVol),:]
volumes = volumes[(volumes > minVol) & (volumes < maxVol)]
kptn_data = kptn_data[kptn_data[:,1].argsort(),:]
kptn_data_log = np.log2(kptn_data[:,2:])

annData = sc.AnnData(X = kptn_data_log)
nn = sc.pp.neighbors(annData, n_neighbors=10)

sc.tl.umap(annData)
sc.pl.umap(annData)
sc.tl.louvain(annData, resolution = 0.1)
sc.pl.umap(annData, color = 'louvain')

annData2 = sc.AnnData(X = kptn_data[:,2:])
nn = sc.pp.neighbors(annData2, n_neighbors=10)

sc.tl.umap(annData2)
sc.pl.umap(annData2)
sc.tl.louvain(annData2, resolution = 0.1)
sc.pl.umap(annData2, color = 'louvain')

