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

for slide in range(35):            

    # Import data:
    kptn_data_all = pd.read_csv(allFiles[slide], sep = '\t' , skiprows = 8, header = 1)
    kptn_data = np.asarray(kptn_data_all[['Position X [µm]', 'Position Y [µm]', 'Nuclei - Intensity Nucleus Alexa 568 Mean', 'Nuclei - Intensity Nucleus Atto 490LS Mean', 'Nuclei - Intensity Nucleus Alexa 488 Mean', 'Nuclei - Intensity Nucleus Alexa 647 Mean', 'Nuclei - Intensity Nucleus Atto 425 Mean']])
    if not os.path.exists('figures/' + slideNames[slide]):
        os.mkdir('figures/' + slideNames[slide])
    if not os.path.exists('data/' + slideNames[slide]):
        os.mkdir('data/' + slideNames[slide])
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
    
    # Make some plots to get overview of data:

    sectionNumber = np.zeros(np.shape(kptn_data_log)[0])
    count = 1
    sectionNumber[0] = count
    for i in range(1, np.shape(kptn_data_log)[0]):
        if abs(kptn_data[i,1] - kptn_data[i-1,1]) > 1000:
            count = count + 1
        sectionNumber[i] = count
        
    # Run GaussianMixture model:
    section = 1
    data = kptn_data[sectionNumber == section,2:]
    n_samples = np.shape(data)[0]
    n_dimensions = np.shape(data)[1]
    n_components = 6
    alpha = np.array((10,5,5,5,10,10))
    
    # Make some informative prior about mean, variance and crosstalk:
    mean_priorMean = np.ones((n_components, n_dimensions))*[np.mean(np.sort(kptn_data_log[:,i])[range(int(np.round(len(kptn_data_log[:,i])/2)))]) for i in range(n_dimensions)]
    mean_priorSigma = np.ones((n_components, n_dimensions))*0.25
    sigma_priorMean = np.ones((n_components, n_dimensions))*0.25
    sigma_priorSigma = np.ones((n_components, n_dimensions))*0.25
    
    for i in range(n_dimensions):
        mean_priorMean[i,i] += 3
        sigma_priorMean[i,i]  = 1
        sigma_priorSigma[i,i] = 1
        
    spectralSignature_priorMean = np.ones((n_dimensions, n_dimensions))*0.1
    spectralSignature_priorMean[1,4] = 0.25
    spectralSignature_priorMean[4,1] = 0.25 # i.e. expect significant bleedthrough/crosstalk between 490LS and 425
    spectralSignature_priorSigma = np.ones((n_dimensions, n_dimensions))*0.1
    for i in range(n_dimensions):
        spectralSignature_priorMean[i,i] = 1
    spectralSignature_priorMean = spectralSignature_priorMean/np.sum(spectralSignature_priorMean, axis = 0)
        
    
    def logp_normal(mu, sigma, value):
    # log probability of individual samples in multivariate normal with diagonal covariance
        return tt.sum([-1/2*(math.log(2*math.pi) + tt.log(sigma[i]**2)) - ((value-mu[i])**2)/(2 * (sigma[i]**2)) for i in range(2)])
    
    # Log likelihood of Gaussian mixture distribution with diagonal covariance
    def logp_gmix(mus, pi, sigmas):
        def logp_(value):
            logps = [tt.log(pi[i]) + logp_normal(mus[i], sigmas[i], value)
                     for i in range(3)]
            return tt.sum(logsumexp(tt.stacklists(logps), axis=0))
        return logp_

        with pm.Model() as model:
            w = pm.Dirichlet('w', alpha)
            mus = pm.Normal('mu', mu = mean_priorMean, sigma = mean_priorSigma, shape = (n_components, n_dimensions))
            sigmas = pm.Gamma('sigma', mu = sigma_priorMean, sigma = sigma_priorSigma, shape = (n_components, n_dimensions))
            c = pm.Normal('c', mu = spectralSignature_priorMean, sigma = spectralSignature_priorSigma, shape = (n_dimensions, n_dimensions))
            prior = sample_prior(samples = 1000)
            data_corrected = tt.log(tt.dot(data,tt.inv(c)))
            x = pm.DensityDist('x', logp_gmix(mus, w, sigmas), observed=data_corrected)
            
            # Plot prior for some parameters:
            f = plt.figure()
            plt.hist(prior['mu'][:,:,0])
            plt.show()
            f.savefig("figures/" + slideNames[slide] + "/" + "muPriorSection" + str(section) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 
            
            f = plt.figure()
            plt.hist(prior['taus_0'][:,1])
            plt.show()
            f.savefig("figures/" + slideNames[slide] + "/" + "sigmaPriorSection" + str(section) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 

            f = plt.figure()
            plt.hist(prior['w'])
            plt.show()
            f.savefig("figures/" + slideNames[slide] + "/" + "wPriorSection" + str(section) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 

            # Fit:
            with model:
                advi_fit = pm.fit(n=500, obj_optimizer=pm.adagrad(learning_rate=1e-1), method = 'advi')
                
            # Sample:
            with model:
                %time hmc_trace = pm.sample(draws=20, tune=50, cores=15)

            # Show results advi:
            f = plt.figure()
            advi_elbo = pd.DataFrame(
                {'log-ELBO': -np.log(advi_fit.hist),
                 'n': np.arange(advi_fit.hist.shape[0])})
            _ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
            f.savefig("figures/" + slideNames[slide] + "/" + "adviElbo_Section" + str(section) + "_channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 
            advi_trace = advi_fit.sample(10000)
            pm.summary(advi_trace, include_transformed=False)
            # Plot of all component distributions in each channel
            f, axis = plt.subplots(n_components,n_dimensions, figsize=(n_components*2.5,n_dimensions*2.5))
            plt.rcParams['axes.titlesize'] = 10
            plt.rcParams['axes.facecolor'] = 'white'
            dotSize = 0.5
            colours = ('gold', 'pink','green', 'red', 'blue')
            x_min = 6
            x_max = 12
            x = np.linspace(x_min, x_max, 100)
            for i in range(n_components):
                for j in range(n_dimensions):
                    axis[i,j].plot(x, scipy.stats.norm.pdf(x,np.mean(advi_trace.get_values('mu')[:,i,j]), np.mean(advi_trace.get_values('sigma')[:,i,j])), color=colours[j])
            mean_posteriorMean = np.zeros((n_components,n_dimensions))
            for i in range(n_components):
                for j in range(n_dimensions):
                    mean_posteriorMean[i,j] = np.mean(advi_trace.get_values('mu')[:,i,j])
                    
                    
            # Show results hmc:
            f, axis = plt.subplots(n_components,n_dimensions, figsize=(n_components*2.5,n_dimensions*2.5))
            plt.rcParams['axes.titlesize'] = 10
            plt.rcParams['axes.facecolor'] = 'white'
            dotSize = 0.5
            colours = ('gold', 'pink','green', 'red', 'blue')
            x_min = 6
            x_max = 12
            x = np.linspace(x_min, x_max, 100)
            for i in range(n_components):
                for j in range(n_dimensions):
                    axis[i,j].plot(x, scipy.stats.norm.pdf(x,np.mean(hmc_trace.get_values('mu')[:,i,j]), np.mean(hmc_trace.get_values('sigma')[:,i,j])), color=colours[j])
            mean_posteriorMean = np.zeros((n_components,n_dimensions))
            for i in range(n_components):
                for j in range(n_dimensions):
                    mean_posteriorMean[i,j] = np.mean(hmc_trace.get_values('mu')[:,i,j])
                    

            # Save trace means:
            advi_mus = np.array([[np.mean(advi_trace.get_values('mu')[:,i,j]) for i in range(n_components)] for j in range(n_dimensions)])
            advi_sigmas = np.array([[np.mean(advi_trace.get_values('sigma')[:,i,j]) for i in range(n_components)] for j in range(n_dimensions)])
            advi_w = np.array([np.mean(advi_trace.get_values('w')[:,i]) for i in range(n_components)])
            advi_data = {"advi_mu": advi_mus,
                         "advi_sigma": advi_sigmas,
                         "advi_w": advi_w}
            pickle_out = open("data/" + slideNames[slide] + '_AdviFitResults.pickle',"wb")
            pickle.dump(advi_data, pickle_out)
            pickle_out.close()
            

            # Calculate class membership, by using advi_trace and logp_normal function:                            
            confidenceThreshold = 0.66
            class0LogProb = [logp_normal(np.mean(advi_trace.get_values('mu')[:,0]), np.mean(advi_trace.get_values('sigma')[:,0]) , data[k,channel]) for k in np.where(sectionNumber == section)]
            class1LogProb = [logp_normal(np.mean(advi_trace.get_values('mu')[:,1]), np.mean(advi_trace.get_values('sigma')[:,1]) , data[k,channel]) for k in np.where(sectionNumber == section)]
            normalizedProbs = [exp_normalize(np.array((class0LogProb[0][i], class1LogProb[0][i]))) for i in range(len(class0LogProb[0]))]
            maxProbs = [max(normalizedProbs[i]) for i in range(len(normalizedProbs))]
            classMembership = [np.argmax(normalizedProbs[i]) for i in range(len(normalizedProbs))]
            confidentClass = [2 if maxProbs[i] < confidenceThreshold else classMembership[i] for i in range(len(classMembership))]

            # Class membership probability:
            pickle_out = open("data/" + slideNames[slide] + "Probability-" + celltypeOrder[channel] + '.pickle',"wb")
            pickle.dump(normalizedProbs, pickle_out)
            pickle_out.close()

            ### Plot results:

            # Histograms:
            if sum(np.array(confidentClass) == 1) > 0:
                boundary1 = min(data[sectionNumber == section,channel][np.array(confidentClass) == 1])
            else:
                boundary1 = np.inf
            if sum(np.array(confidentClass) == 2) > 0:
                boundary2 = min(data[sectionNumber == section,channel][np.array(confidentClass) == 2])
            else:
                boundary2 = 0
            fig = plt.figure()
            fig, ax = plt.subplots()
            N, bins, patches = ax.hist(data[sectionNumber == section,channel], edgecolor='white', linewidth=1, bins = 100)
            for i in range(0, len(patches)):
                if bins[i] < boundary2:
                    patches[i].set_facecolor('b')   
                elif bins[i] < boundary1:
                    patches[i].set_facecolor('black')
                else:
                    patches[i].set_facecolor('r')
            plt.gca().set_title('Log Intensity and Classification Channel ' + channelOrder[channel])
            plt.show()
            fig.savefig("figures/" + slideNames[slide] + "/" + "HistogramIntensityAndClassification" + "Section" + str(section) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(fig) 
            
            # Scatterplots:
            colours = np.repeat('black', sum(sectionNumber == section))                            
            if sum(np.array(confidentClass) == 1) > 0:
                colours[np.array(confidentClass) == 1] = 'red'  

            fig = plt.figure()                            
            plt.scatter(kptn_data[sectionNumber == section,0], np.exp(data[sectionNumber == section,channel]), c = colours, s = 0.1)
            plt.gca().set_title('Intensity and Classification Channel ' + channelOrder[channel])
            plt.show()
            fig.savefig("figures/" + slideNames[slide] + "/" + "ScatterPlotIntensityAndClassification" + "Section" + str(section) + "channel" + channelOrder[channel] + ".png", bbox_inches='tight')  
            plt.close(fig) 
            
            fig = plt.figure()                             
            plt.scatter(kptn_data[sectionNumber == section,0], data[sectionNumber == section,channel], c = colours, s = 0.1)                         
            plt.gca().set_title('Log Intensity and Classification Channel ' + channelOrder[channel])
            plt.show()
            fig.savefig("figures/" + slideNames[slide] + "/" + "ScatterPlotLOGIntensityAndClassification" + "Section" + str(section) + "channel" + channelOrder[channel] + ".png", bbox_inches='tight')
            plt.close(fig) 
            
            # Slide location of each cell type (including unclassified):

            fig = plt.figure()   
            plt.scatter(kptn_data[sectionNumber == section,0][np.array(confidentClass) == 1], kptn_data[sectionNumber == section,1][np.array(confidentClass) == 1], s = 0.05)
            plt.gca().set_title('Nuclei Positive Classification Slide  ' + str(slide) + " Section " + str(section) + " Channel " + channelOrder[channel] + ".png")
            plt.show()
            fig.savefig("figures/" + slideNames[slide] + "/" + "PositiveClassificationPosition" + str(slide) + "section" + str(section) + "channel" + channelOrder[channel] + ".png", bbox_inches='tight')  
            plt.close(fig) 