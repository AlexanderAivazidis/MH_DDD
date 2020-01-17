from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.stats as stats
import math

%matplotlib inline

import functools
import os
os.chdir('/home/jovyan/MH_DDD/')

import matplotlib.pyplot as plt; plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import seaborn as sns; sns.set_context('notebook')

# Import data:
relevantFeatures = ( 'Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Cell Alexa 488 Mean', 'Nuclei - Intensity Cell Alexa 568 Mean', 'Nuclei - Intensity Cell Alexa 647 Mean', 'Nuclei - Intensity Cell Atto 425 Mean', 'Nuclei - Intensity Cell Atto 490LS Mean')
#relevantFeatures = ('Nuclei - Intensity Cell Alexa 488 Maximum', 'Nuclei - Intensity Cell Alexa 568 Maximum', 'Nuclei - Intensity Cell Alexa 647 Maximum',
#'Nuclei - Intensity Cell Atto 425 Maximum', 'Nuclei - Intensity Cell Atto 490LS Maximum')
header = []
indices_I_want = {9}
for i, row in enumerate(open('/home/jovyan/data/KptnMouse/RNAscope/Objects_Population - Nuclei.txt')):
    if i in indices_I_want:
        header.append(row)
header = np.asarray(header[0].split('\t'))
relevantColumns = np.isin(header, relevantFeatures)
relevantColumns = np.asarray(range(len(header)))[relevantColumns]
kptn_data = np.loadtxt('/home/jovyan/data/KptnMouse/RNAscope/Objects_Population - Nuclei.txt', skiprows = 10, usecols = relevantColumns, delimiter = '\t')

kptn_data = kptn_data[kptn_data[:,2].argsort(),:]

kptn_data_log = np.log2(kptn_data[:,2:])

# Plot image for NeuN/Plp1
hist, xedges, yedges = np.histogram2d(kptn_data_log[:,3], kptn_data_log[:,4], bins = 100)
H = hist.T
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(131, title='Frequency of NeuN/Plp1 combinations')
plt.xlabel('NeuN')
plt.ylabel('Plp1')
im = plt.imshow(H, interpolation = 'bilinear', origin='low', extent=[min(xedges[0], yedges[0]), max(xedges[-1], yedges[-1]), min(xedges[0], yedges[0]),  max(xedges[-1], yedges[-1])])
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.show(im)

# Plot image for NeuN/Slc1a3
hist, xedges, yedges = np.histogram2d(kptn_data_log[:,1], kptn_data_log[:,3], bins = 100)
H = hist.T
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(131, title='Frequency of NeuN/Slc1a3 combinations')
plt.xlabel('Slc1a3')
plt.ylabel('NeuN')
im = plt.imshow(H, interpolation = 'bilinear', origin='low', extent=[min(xedges[0], yedges[0]), max(xedges[-1], yedges[-1]), min(xedges[0], yedges[0]),  max(xedges[-1], yedges[-1])])
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.show(im)

#### Run GaussianMixture model with sklearn:
from matplotlib.colors import LogNorm
from sklearn import mixture
import matplotlib.cm as cm
from matplotlib.pyplot import figure

# fit a Bayesian Gaussian Mixture Model
clf = mixture.BayesianGaussianMixture(n_components = 6, covariance_type='full', max_iter = 1000, weight_concentration_prior = 1,
                                      weight_concentration_prior_type = 'dirichlet_process')
clf.fit(kptn_data_log)
labels = clf.predict(kptn_data_log)
probs = clf.predict_proba(kptn_data_log)

# display class predictions by the model in 2D:
colors = cm.Set1
plt.scatter(kptn_data_log[:, 3], kptn_data_log[:, 4], .8, color = colors(labels), alpha = 0.1)

plt.title('Class predictions by GMM')
plt.axis('tight')
axes = plt.gca()
axes.set_xlim([6,10])
axes.set_ylim([6,12])
plt.xlabel('NeuN')
plt.ylabel('Plp1')
plt.show()

# Display intensity distributions for each component and plot covariance matrix as image
meanArray = clf.means_
covarianceArray = clf.covariances_
channels = ('Pdgfra - OPC', 'Slc1a3 - Astro', 'Gad1 - GABA', 'NeuN - Neurons', 'Plp1 - Oligos')
fig = plt.figure(figsize=(25,10))
for i in range(np.shape(meanArray)[0]):
    plt.subplot(2, 3, i+1)
    for j in range(np.shape(meanArray)[1]):
        sigma = math.sqrt(covarianceArray[i,j,j])
        mu = meanArray[i,j]
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), label = channels[j])
        axes = plt.gca()
        axes.legend()
        axes.set_ylim([0,10])