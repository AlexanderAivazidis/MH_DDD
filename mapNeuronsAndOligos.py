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
#%env THEANO_FLAGS=device=cpu,floatX=float32,exception_verbosity='high'
import theano
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
from theano.tensor.nlinalg import det
import seaborn as sns
from pymc3Utils import sample_prior, exp_normalize
import pandas as pd
from fnmatch import fnmatch
import pickle
import math as math
import scipy.stats
from scipy.stats import gaussian_kde

# Run a Multivariate Gaussian Mixture Model for mapping just Neurons and Oligondendrocytes:

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

chunkID = 7
chunkSize = 6
chunks = [range(len(allFiles))[i:i + chunkSize] for i in range(0, len(allFiles), chunkSize)]              
        
celltypeOrder = np.array(('Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron'))
for slide in chunks[int(chunkID)]:     
    for hemisphere in np.array((1,2)):
        
        file = open("data/cortexData/" + slideNames[slide] + 'Section' + sectionID[slide] + 'cortexData_.pickle', 'rb')
        cortexData = pickle.load(file)
        file.close()

        # Use only cortex:

        subset = cortexData['Hemisphere'] == hemisphere
        cortexData = cortexData[subset]
        subset = [cortexData['x_dash-coordinate'].iloc[i] > 0.33 and cortexData['x_dash-coordinate'].iloc[i] < 0.66 for i in range(len(cortexData['x_dash-coordinate']))]
        cortexData = cortexData[subset]
        n_samples = np.shape(cortexData)[0]
        colnames = [cortexData.columns[i] for i in (16,17,18,19,20)]
        data = np.log2(np.array(cortexData.loc[:,[cortexData.columns[i] for i in (16,17,18,19,20)]]))

        n_dimensions = len(colnames)
        n_components = len(colnames) + 1

#         # Plot data:

#         f, axis = plt.subplots(n_dimensions,n_dimensions, figsize=(n_dimensions*n_dimensions,n_dimensions*n_dimensions))
#         plt.rcParams['axes.titlesize'] = 10
#         plt.rcParams['axes.facecolor'] = 'white'
#         dotSize = 0.5
#         for i in range(n_dimensions):
#             for j in range(n_dimensions):
#                 axis[i,j].scatter(data[:, i], data[:, j], c='g', alpha = 0.01)
#                 axis[i,j].set_xlabel(colnames[i])
#                 axis[i,j].set_ylabel(colnames[j])

#         # Run 

        # Make some informative priors:
        signalMean_priorMean = np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])*0.9)),len(data[:,i]))]) for i in range(n_dimensions)])
        backgroundMean_priorMean =  np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])/2)))]) for i in range(n_dimensions)])
        signalMean_priorSD = np.repeat(0.1, n_dimensions)
        backgroundMean_priorSD = np.repeat(0.1, n_dimensions)
        alpha = np.repeat(1, n_components)

        signalSD_priorMean = np.array((0.5,0.5, 0.5, 0.5, 0.25))
        signalSD_priorSD = np.repeat(0.1, n_dimensions)

        backgroundSD_priorMean = np.repeat(0.1, n_dimensions)
        backgroundSD_priorSD = np.repeat(0.1, n_dimensions)

        componentSD_priorMean = np.empty((n_components, n_dimensions))
        for i in range(n_components):
            for j in range(n_dimensions):
                componentSD_priorMean[i,j] = backgroundSD_priorMean[j]
                if i == j:
                    componentSD_priorMean[i,j] = signalMean_priorMean[i]


        componentSD_priorSD = np.empty((n_components, n_dimensions))
        for i in range(n_components):
            for j in range(n_dimensions):
                componentSD_priorSD[i,j] = backgroundSD_priorSD[j]
                if i == j:
                    componentSD_priorSD[i,j] = signalSD_priorSD[i]

        ## Plot prior:

#         # Plot predicted and actual distribution of intensities for each channel:
#         f, axis = plt.subplots(n_dimensions,1, figsize=(10,10*n_dimensions))
#         dotSize = 0.5
#         colours = ('gold', 'pink','green', 'red', 'blue')
#         x_min = 0
#         x_max = np.max(data)
#         x = np.linspace(x_min, x_max, 100)
#         for i in range(n_dimensions):
#             axis[i].plot(x, scipy.stats.norm.pdf(x,signalMean_priorMean[i], signalSD_priorMean[i])*1/n_components, color='black', linewidth = 5)

#             axis[i].plot(x, scipy.stats.norm.pdf(x,backgroundMean_priorMean[i], backgroundSD_priorMean[i])*(1-1/n_components), color='black', linewidth = 5)
#             axis[i].hist(data[:,i], density = True, bins = 50)    

#         ## Model:

        # Log likelihood of normal distribution
        def logp_normal(mu, tau, value):
            # log probability of individual samples
            k = tau.shape[0]
            delta = lambda mu: value - mu
            return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                                 (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

        # Log likelihood of Gaussian mixture distribution
        def logp_gmix(mus, pi, taus, n_components):
            def logp_(value):        
                logps = [tt.log(pi[i]) + logp_normal(mus[i,:], taus[i], value) for i in range(n_components)]
                return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))
            return logp_

        # Sparse model with diagonal covariance:
        with pm.Model() as model:

            # Weights of each component:
            w = Dirichlet('w', a=pm.floatX(alpha), shape=(n_components,))

            # Impose sparse structure onto mean with off-diagonal elements all being the same, because background should be the same throughout.
            mus_signal = MvNormal('mus_signal', mu=pm.floatX(signalMean_priorMean), tau=pm.floatX(np.eye(n_dimensions)/signalMean_priorSD**2), shape=n_dimensions)
            mus_background = MvNormal('mus_background', mu=pm.floatX(backgroundMean_priorMean), tau=pm.floatX(np.eye(n_dimensions)/backgroundMean_priorSD**2), shape=n_dimensions)
            mus = tt.fill_diagonal(tt.reshape(tt.tile(mus_background, n_components), (n_components,n_dimensions)),0) + tt.eye(n_components, n_dimensions) * mus_signal

            # Impose structure for covariance as well, with off-diagonal elements being zero, just because that model is easier to fit. 
            sigmas_signal = pm.Gamma('sigmas_signal', mu=pm.floatX(signalSD_priorMean), sd=pm.floatX(signalSD_priorSD), shape=n_dimensions)
            sigmas_background = pm.Gamma('sigmas_background', mu=pm.floatX(backgroundSD_priorMean), sd=pm.floatX(backgroundSD_priorSD), shape=n_dimensions)
            sigmas = tt.fill_diagonal(tt.reshape(tt.tile(sigmas_background, n_components), (n_components,n_dimensions)),0) + tt.eye(n_components, n_dimensions) * sigmas_signal
            covs = [tt.zeros((n_dimensions, n_dimensions)) + tt.eye(n_dimensions, n_dimensions) * sigmas[i,:] for i in range(n_components)]
            taus = [tt.nlinalg.matrix_inverse(covs[i])**2 for i in range(n_components)]

            # Gaussian Mixture Model:
            x = DensityDist('x', logp_gmix(mus, w, taus, n_components), observed=pm.floatX(data))

        with model:
            advi_fit = pm.fit(n=100000, obj_optimizer=pm.adagrad(learning_rate=0.01))  

        advi_trace = advi_fit.sample(10000)    

        advi_summary = pm.summary(advi_trace, include_transformed=False)

        pickle_out = open("advi_summaries/advi_summary_slide" + str(slide) + 'hemisphere_' + str(hemisphere) + ".pickle","wb")
        pickle.dump(advi_summary, pickle_out)
        pickle_out.close()

    #     advi_elbo = pd.DataFrame(
    #         {'log-ELBO': -np.log(advi_fit.hist),
    #          'n': np.arange(advi_fit.hist.shape[0])})
    #     _ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
    #     plt.savefig('books_read.png')

    #     pickle_in = open("advi_summary.pickle","rb")
    #     advi_summary = pickle.load(pickle_in)

    #     pickle.load('advi_summary.pickle')

    #     ## Plot the predicted mean and variance for each component vs. the actual one and also plot the inferred covariance matrices:

    #     meanMatrix = np.empty((n_components, n_dimensions))
    #     for i in range(n_components):
    #         for j in range(n_dimensions):
    #             meanMatrix[i,j] = advi_summary['mean']['mus_background__' + str(j)]
    #             if i == j:
    #                 meanMatrix[i,j] = advi_summary['mean']['mus_signal__' + str(j)]

    #     sigmaMatrix = np.zeros((n_components,n_dimensions,n_dimensions))
    #     for i in range(n_components):
    #         for j in range(n_dimensions):
    #             sigmaMatrix[i,j,j] = advi_summary['mean']['sigmas_background__' + str(j)]
    #             if i == j:
    #                 sigmaMatrix[i,j,j] = advi_summary['mean']['sigmas_signal__' + str(j)]

    #     meanVector = np.empty(n_dimensions)
    #     for i in range(n_dimensions):
    #         meanVector[i] = advi_summary['mean']['mus_signal__' + str(i)]

    #     sigmaVector = np.empty(n_dimensions)
    #     for i in range(n_dimensions):
    #         sigmaVector[i] = advi_summary['mean']['sigmas_signal__' + str(i)]

    #     weightsVector = np.empty((n_components))
    #     for i in range(n_components):
    #         weightsVector[i] = advi_summary['mean']['w__' + str(i)]

    #     # Plot predicted and actual distribution of intensities for each channel:
    #     f, axis = plt.subplots(n_dimensions,1, figsize=(10,n_dimensions*10))
    #     dotSize = 0.5
    #     colours = ('gold', 'pink','green', 'red', 'blue')
    #     x_min = 0
    #     x_max = np.max(data)
    #     x = np.linspace(x_min, x_max, 100)
    #     for i in range(n_dimensions):
    #         axis[i].plot(x, scipy.stats.norm.pdf(x,meanVector[i], sigmaVector[i])*weightsVector[i], color='black', linewidth = 5)
    #         axis[i].plot(x, scipy.stats.norm.pdf(x,advi_summary['mean']['mus_background__' + str(i)], 
    #         advi_summary['mean']['sigmas_background__' + str(i)])*sum(weightsVector[np.arange(len(weightsVector))!=i]), color='black', linewidth = 5)

    #         axis[i].hist(data[:,i], density = True, bins = 50)