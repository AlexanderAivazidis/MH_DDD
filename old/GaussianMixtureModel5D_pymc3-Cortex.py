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
#%env THEANO_FLAGS=device=cpu,floatX=float32
import theano
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
from theano.tensor.nlinalg import det
import seaborn as sns
from pymc3Utils import sample_prior, exp_normalize
from pymc3Distributions import logp_normal
import pandas as pd
from fnmatch import fnmatch
import pickle
import math as math
import scipy.stats

# Get all slides that have the cortex, plus cortical depth segmented:
root = 'data/cortexData/'
pattern = "*cortexData_.pickle"
allFiles = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            allFiles.append(os.path.join(path, name))
            
for i in range(len(allFiles)): 
    
    print(i)
    # Import nuclei positions, channel intensities, region classification, cortical depth, radial distance:
    file = open(allFiles[i], 'rb')
    cortexData = pickle.load(file)
    file.close()
        
    # Run GaussianMixture model:
    n_samples = np.shape(cortexData)[0]
    data = np.array(cortexData[cortexData.columns[16:21]])
    n_dimensions = 5
    n_components = 6
    alpha = np.array((5,2,1,1,10,5))
    
    # Make some informative prior about mean, variance:
    signalMean_priorMean = np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])*0.95)),len(data[:,i]))]) for i in range(n_dimensions)])
    backgroundMean_priorMean =  np.array([np.mean(np.sort(data[:,i])[range(int(np.round(len(data[:,i])/2)))]) for i in range(n_dimensions)])
    signalMean_priorSD = np.array((1000,1000,1000,1000,1000))
    backgroundMean_priorSD = np.array((50,50,50,50,50))
    signalSigma_priorMean = np.array((100,100,100,100,100))
    signalSigma_priorSD = np.array((1000,1000,1000,1000,1000))
    backgroundSigma_priorMean = np.array((10,10,10,10,10))
    backgroundSigma_priorSD = np.array((50,50,50,50,50))
    neunGABAMean_priorMean = signalMean_priorMean[4]/2
    neunGABAMean_priorSD = 250
    neunGABASD_priorMean = 100
    neunGABASD_priorSD = 100
        
    def logp_normal(mu, sigma, value):
    # log probability of individual samples in multivariate normal with diagonal covariance (i.e. just the summed logP of multiple independent normals)
        return tt.sum([-1/2*(math.log(2*math.pi) + tt.log(sigma[i]**2)) - ((value[i]-mu[i])**2)/(2 * (sigma[i]**2)) for i in range(n_dimensions)])
    
    # Log likelihood of Gaussian mixture distribution with diagonal covariance
    def logp_gmix(mus, pi, sigmas):
        def logp_(value):
            logps = [tt.log(pi[i]) + logp_normal(mus[i,:], sigmas[i,:], value) for i in range(n_components)]
            return tt.sum(logsumexp(tt.stacklists(logps), axis=0))
        return logp_

    with pm.Model() as model:
        w = pm.Dirichlet('w', alpha)
        #mus_signal = pm.Normal('mu_signal', mu = np.append(signalMean_priorMean, neunGABAMean_priorMean), sigma = np.append(signalMean_priorSD, neunGABAMean_priorSD), shape = (n_dimensions+1))
        mus_signal = pm.Normal('mu_signal', mu = signalMean_priorMean, sigma = signalMean_priorSD, shape = (n_dimensions))
        mus_background = pm.Normal('mu_background', mu = backgroundMean_priorMean, sigma = backgroundMean_priorSD, shape = (n_dimensions))
        #sigmas_signal = pm.Gamma('sigma_signal', mu = np.append(signalSigma_priorMean, neunGABASD_priorMean), sigma = np.append(signalSigma_priorSD, neunGABASD_priorSD), shape = (n_dimensions+1))
        sigmas_signal = pm.Gamma('sigma_signal', mu = signalSigma_priorMean, sigma = signalSigma_priorSD, shape = (n_dimensions))
        sigmas_background = pm.Gamma('sigma_background', mu = backgroundSigma_priorMean, sigma = backgroundSigma_priorSD, shape = (n_dimensions))
        mus = np.array(((mus_signal[0], mus_background[1], mus_background[2], mus_background[3], mus_background[4]), 
                                                (mus_background[0], mus_signal[1], mus_background[2], mus_background[3], mus_background[4]),
                                                (mus_background[0], mus_background[1], mus_signal[2], mus_background[3], mus_background[4]),
                                                (mus_background[0], mus_background[1], mus_background[2], mus_signal[3], mus_background[4]),
                                                (mus_background[0], mus_background[1], mus_background[2], mus_background[3], mus_signal[4]),
                                                (mus_background[0], mus_background[1], mus_background[2], mus_background[3], mus_background[4])))
        sigmas = np.array(((sigmas_signal[0], sigmas_background[1], sigmas_background[2], sigmas_background[3], sigmas_background[4]), 
                                                (sigmas_background[0], sigmas_signal[1], sigmas_background[2], sigmas_background[3], sigmas_background[4]),
                                                (sigmas_background[0], sigmas_background[1], sigmas_signal[2], sigmas_background[3], sigmas_background[4]),
                                                (sigmas_background[0], sigmas_background[1], sigmas_background[2], sigmas_signal[3], sigmas_background[4]),
                                                (sigmas_background[0], sigmas_background[1], sigmas_background[2], sigmas_background[3], sigmas_signal[4]),
                                                (sigmas_background[0], sigmas_background[1], sigmas_background[2], sigmas_background[3], sigmas_background[4])))        
        prior = sample_prior(samples = 1000)
        x = pm.DensityDist('x', logp_gmix(mus, w, sigmas), observed=data.T)
    
    def logp_normal(mu, sigma, value):
# log probability of individual samples in multivariate normal with diagonal covariance (i.e. just the summed logP of multiple independent normals)
        return tt.sum([-1/2*(math.log(2*math.pi) + tt.log(sigma[i]**2)) - ((value[i]-mu[i])**2)/(2 * (sigma[i]**2)) for i in range(2)])
    
    # Log likelihood of Gaussian mixture distribution with diagonal covariance
    def logp_gmix(mus, pi, sigmas):
        def logp_(value):
            logps = [tt.log(pi[i]) + logp_normal(mus[i,:], sigmas[i,:], value) for i in range(2)]
            return tt.sum(logsumexp(tt.stacklists(logps), axis=0))
        return logp_
    
    alpha = np.array((1,2))
    with pm.Model() as model:
        w = pm.Dirichlet('w', alpha)
        #mus_signal = pm.Normal('mu_signal', mu = np.append(signalMean_priorMean, neunGABAMean_priorMean), sigma = np.append(signalMean_priorSD, neunGABAMean_priorSD), shape = (n_dimensions+1))
        mus_signal = pm.Normal('mu_signal', mu = signalMean_priorMean[1:3], sigma = signalMean_priorSD[1:3], shape = (n_dimensions-3))
        mus_background = pm.Normal('mu_background', mu = backgroundMean_priorMean[1:3], sigma = backgroundMean_priorSD[1:3], shape = (n_dimensions-3))
        #sigmas_signal = pm.Gamma('sigma_signal', mu = np.append(signalSigma_priorMean, neunGABASD_priorMean), sigma = np.append(signalSigma_priorSD, neunGABASD_priorSD), shape = (n_dimensions+1))
        sigmas_signal = pm.Gamma('sigma_signal', mu = signalSigma_priorMean[1:3], sigma = signalSigma_priorSD[1:3], shape = (n_dimensions-3))
        sigmas_background = pm.Gamma('sigma_background', mu = backgroundSigma_priorMean[1:3], sigma = backgroundSigma_priorSD[1:3], shape = (n_dimensions-3))
        mus = np.array(((mus_signal[0], mus_background[1]), 
                                                (mus_background[0], mus_signal[1])))
        sigmas = np.array(((sigmas_signal[0], sigmas_background[1]), 
                                                (sigmas_background[0], sigmas_signal[1])))        
        prior = sample_prior(samples = 1000)
        x = pm.DensityDist('x', logp_gmix(mus, w, sigmas), observed=data[:,1:3])
    
    # Plot prior for some parameters:

#     plt.hist(prior['mu'][:,:,0])
#     plt.show()

#     plt.hist(prior['taus_0'][:,1])
#     plt.show()

    plt.hist(prior['sigma_signal'])
    plt.show()

    # Fit:
    with model:
        advi_fit = pm.fit(n=5000, obj_optimizer=pm.adagrad(learning_rate=1e-1), method = 'advi')

    # Sample:
    with model:
        %time hmc_trace = pm.sample(draws=50, tune=1000, cores=16, target_accept=0.99)

    # Show results advi:
    f = plt.figure()
    advi_elbo = pd.DataFrame(
        {'log-ELBO': -np.log(advi_fit.hist),
         'n': np.arange(advi_fit.hist.shape[0])})
    _ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
    advi_trace = advi_fit.sample(10000)
    pm.summary(advi_trace, include_transformed=False)
    # Plot predicted and actual distribution of intensities for each channel:
    f, axis = plt.subplots(1,n_dimensions, figsize=(n_dimensions*10,10))
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.facecolor'] = 'white'
    dotSize = 0.5
    colours = ('gold', 'pink','green', 'red', 'blue')
    x_min = 0
    for j in range(n_dimensions):
        x_max = np.max(data[:,j])
        x = np.linspace(x_min, x_max, 100)
        axis[j].plot(x, scipy.stats.norm.pdf(x,np.mean(advi_trace.get_values('mu_signal')[:,j]), np.mean(advi_trace.get_values('sigma_signal')[:,j]))*np.mean(advi_trace.get_values('w')[:,j]), color=colours[j], linewidth = 5)
        axis[j].plot(x, scipy.stats.norm.pdf(x,np.mean(advi_trace.get_values('mu_background')[:,j]), np.mean(advi_trace.get_values('sigma_background')[:,j]))*np.sum([np.mean(advi_trace.get_values('w')[:,i]) for i in range(n_components) if i != j]), color=colours[j], linewidth = 5)
        axis[j].hist(data[:,j], density = True, bins = 50)
    # Without weighting my mixture weight:
    # Plot predicted and actual distribution of intensities for each channel:
    f, axis = plt.subplots(n_dimensions, figsize=(10,n_dimensions*10))
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.facecolor'] = 'white'
    dotSize = 0.5
    colours = ('gold', 'pink','green', 'red', 'blue')
    x_min = 0
    for j in range(n_dimensions):
        x_max = np.max(data[:,j])
        x = np.linspace(x_min, x_max, 100)
        axis[j].plot(x, scipy.stats.norm.pdf(x,np.mean(advi_trace.get_values('mu_signal')[:,j]), np.mean(advi_trace.get_values('sigma_signal')[:,j])), color=colours[j])
        axis[j].plot(x, scipy.stats.norm.pdf(x,np.mean(advi_trace.get_values('mu_background')[:,j]), np.mean(advi_trace.get_values('sigma_background')[:,j])), color=colours[j])
        axis[j].hist(data[:,j], density = True, bins = 50)  
        
#             # Show results hmc:
#             f, axis = plt.subplots(n_components,n_dimensions, figsize=(n_components*2.5,n_dimensions*2.5))
#             plt.rcParams['axes.titlesize'] = 10
#             plt.rcParams['axes.facecolor'] = 'white'
#             dotSize = 0.5
#             colours = ('gold', 'pink','green', 'red', 'blue')
#             x_min = 6
#             x_max = 12
#             x = np.linspace(x_min, x_max, 100)
#             for i in range(n_components):
#                 for j in range(n_dimensions):
#                     axis[i,j].plot(x, scipy.stats.norm.pdf(x,np.mean(hmc_trace.get_values('mu')[:,i,j]), np.mean(hmc_trace.get_values('sigma')[:,i,j])), color=colours[j])
#             mean_posteriorMean = np.zeros((n_components,n_dimensions))
#             for i in range(n_components):
#                 for j in range(n_dimensions):
#                     mean_posteriorMean[i,j] = np.mean(hmc_trace.get_values('mu')[:,i,j])
    
    # Construct the matrix of fitted component mean and variances in each channel:
    
    signalMean = [np.mean(advi_trace.get_values('mu_signal')[:,j]) for j in range(5)]
    backgroundMean = [np.mean(advi_trace.get_values('mu_background')[:,j]) for j in range(5)]
    signalSigma = [np.mean(advi_trace.get_values('sigma_signal')[:,j]) for j in range(5)]
    backgroundSigma = [np.mean(advi_trace.get_values('sigma_background')[:,j]) for j in range(5)]
    
    meanMatrix = np.array(((signalMean[0], backgroundMean[1], backgroundMean[2], backgroundMean[3], backgroundMean[4]),
                          (backgroundMean[0], signalMean[1], backgroundMean[2], backgroundMean[3], backgroundMean[4]),
                          (backgroundMean[0], backgroundMean[1], signalMean[2], backgroundMean[3], backgroundMean[4]),
                          (backgroundMean[0], backgroundMean[1], backgroundMean[2], signalMean[3], backgroundMean[4]),
                          (backgroundMean[0], backgroundMean[1], backgroundMean[2], backgroundMean[3], signalMean[4]),
                          (backgroundMean[0], backgroundMean[1], backgroundMean[2], backgroundMean[3], backgroundMean[4])))
    
    sigmaMatrix = np.array(((signalSigma[0], backgroundSigma[1], backgroundSigma[2], backgroundSigma[3], backgroundSigma[4]),
                          (backgroundSigma[0], signalSigma[1], backgroundSigma[2], backgroundSigma[3], backgroundSigma[4]),
                          (backgroundSigma[0], backgroundSigma[1], signalSigma[2], backgroundSigma[3], backgroundSigma[4]),
                          (backgroundSigma[0], backgroundSigma[1], backgroundSigma[2], signalSigma[3], backgroundSigma[4]),
                          (backgroundSigma[0], backgroundSigma[1], backgroundSigma[2], backgroundSigma[3], signalSigma[4]),
                          (backgroundSigma[0], backgroundSigma[1], backgroundSigma[2], backgroundSigma[3], backgroundSigma[4])))

#     # Save trace means:
#     advi_mus = np.array([[np.mean(advi_trace.get_values('mu')[:,i,j]) for i in range(n_components)] for j in range(n_dimensions)])
#     advi_sigmas = np.array([[np.mean(advi_trace.get_values('sigma')[:,i,j]) for i in range(n_components)] for j in range(n_dimensions)])
#     advi_w = np.array([np.mean(advi_trace.get_values('w')[:,i]) for i in range(n_components)])
#     advi_data = {"advi_mu": advi_mus,
#                  "advi_sigma": advi_sigmas,
#                  "advi_w": advi_w}
#     pickle_out = open("data/" + slideNames[slide] + '_AdviFitResults.pickle',"wb")
#     pickle.dump(advi_data, pickle_out)
#     pickle_out.close()
    
    def logp_normal2(mu, sigma, value):
    # log probability of individual samples in multivariate normal with diagonal covariance (i.e. just the summed logP of multiple independent normals)
        return sum([-1/2*(math.log(2*math.pi) + math.log(sigma[i]**2)) - ((value[i]-mu[i])**2)/(2 * (sigma[i]**2)) for i in range(n_dimensions)])
    
    # Calculate class membership, by using advi_trace and logp_normal function:                            
    confidenceThreshold = 0.95
    classLogProb = [[logp_normal2(meanMatrix[i,:], sigmaMatrix[i,:] , data[x,:]) for i in range(n_components)] for x in range(len(data[:,0]))]
    normalizedProbs = [exp_normalize(np.array(classLogProb[i])) for i in range(len(classLogProb))]
    maxProbs = [max(normalizedProbs[i]) for i in range(len(normalizedProbs))]
    classMembership = [np.argmax(normalizedProbs[i]) for i in range(len(normalizedProbs))]
    confidentClass = [6 if maxProbs[i] < confidenceThreshold else classMembership[i] for i in range(len(classMembership))]
    np.sum(np.array(confidentClass) == 0)
    np.sum(np.array(confidentClass) == 1)
    np.sum(np.array(confidentClass) == 2)
    np.sum(np.array(confidentClass) == 3)
    np.sum(np.array(confidentClass) == 4)
    np.sum(np.array(confidentClass) == 5)
    np.sum(np.array(confidentClass) == 6)
    
    # Plot classifications in a series of scatterplots:
    
    figureObjects = ('Nuclei', 'Astrocyte', 'Oligodendrocyte', 'GABAergicNeuron', 'OPC', 'Neuron', 'Other', 'Unclassified')
    objectsColours = ('black', 'gold', 'pink','green', 'red', 'blue', 'black','grey')
    channels = ('568', '490LS', '488', '647', '425')
    colourIndex = np.repeat(int(1), len(confidentClass))                            
    for j in range(7):
        colourIndex[np.array(confidentClass) == j] = int(j+1)
    
    f, axis = plt.subplots(2,8, figsize=(40,2*5))
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.facecolor'] = 'white'
    dotSize = 0.5  
    
    axis[0,0].set_title(figureObjects[0] + ' Positions \n' + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0])
    axis[0,0].scatter(cortexData['x-coordinate'], cortexData['y-coordinate'], s = dotSize, color = objectsColours[0])
    
    for j in range(7):
    
        axis[0,j+1].set_title(figureObjects[j+1] + ' Positions \n' + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0])
        axis[0,j+1].scatter(np.array(cortexData['x-coordinate'])[np.array(confidentClass) == j], np.array(cortexData['y-coordinate'])[np.array(confidentClass) == j], s = dotSize, color = objectsColours[j+1])
        if j < 5:
            axis[1,j+1].set_title('Channel ' + channels[j] + ' Intensities and Classification \n' + cortexData['SlideName'][0] + 'Section ' + cortexData['Section'][0])
            axis[1,j+1].scatter(cortexData['x-coordinate'], np.log2(data[:,j]) ,s = dotSize, c = [objectsColours[colourIndex[i]] for i in range(len(colourIndex))])

#     # Class membership probability:
#     pickle_out = open("data/" + slideNames[slide] + "Probability-" + celltypeOrder[channel] + '.pickle',"wb")
#     pickle.dump(normalizedProbs, pickle_out)
#     pickle_out.close()