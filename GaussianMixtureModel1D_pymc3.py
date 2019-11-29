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
import sys

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

chunkID = sys.argv[1]
chunkSize = 5
chunks = [range(len(slideNames))[i:i + chunkSize] for i in range(0, len(slideNames), chunkSize)]              
            
for slide in chunks[int(chunkID)]:            

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

    f = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Nuclei Position Slide ' + str(slide) + ' Before and After Filtering')
    plt.scatter(kptn_data_all['Position X [µm]'], kptn_data_all['Position Y [µm]'], s = 0.05)
    plt.subplot(1, 2, 2)
    plt.scatter(kptn_data[:,0], kptn_data[:,1], s = 0.05)
    plt.show()
    f.savefig("figures/" + slideNames[slide] + "/_NucleiPosition.png", bbox_inches='tight')
    plt.close(f)    # close the figure window
    
    f = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Nuclei Volumes Slide ' + str(slide) + ' Before and After Filtering')
    plt.hist(kptn_data_all['Nuclei - Nucleus Volume [µm³]'], bins = 100)
    plt.subplot(2, 1, 2)
    plt.hist(volumes, bins = 100)
    plt.show()
    f.savefig("figures/" + slideNames[slide] + "/" + "_NucleiVolumes.png", bbox_inches='tight')
    plt.close(f) 
    
    # Make some plots to get overview of data:

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
        pickle_out = open("data/" + slideNames[slide] + 'Section' + str(int(section)) + '_NucleixyPositions.pickle',"wb")
        pickle.dump(nucleiPositions, pickle_out)
        pickle_out.close()

    for section in np.unique(uniqueSectionNumbers):
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        plt.subplot(5, 1, 1)
        plt.hist(kptn_data_log[sectionNumber == section,0], bins = 100)
        plt.gca().set_title('Intensity Distribution Channel ' + channelOrder[0])
        plt.subplot(5, 1, 2)
        plt.hist(kptn_data_log[sectionNumber == section,1], bins = 100)
        plt.gca().set_title('Intensity Distribution Channel ' + channelOrder[1])
        plt.subplot(5, 1, 3)
        plt.hist(kptn_data_log[sectionNumber == section,2], bins = 100)
        plt.gca().set_title('Intensity Distribution Channel ' + channelOrder[2])
        plt.subplot(5, 1, 4)
        plt.hist(kptn_data_log[sectionNumber == section,3], bins = 100)
        plt.gca().set_title('Intensity Distribution Channel ' + channelOrder[3])
        plt.subplot(5, 1, 5)
        plt.hist(kptn_data_log[sectionNumber == section,4], bins = 100)
        plt.gca().set_title('Intensity Distribution Channel ' + channelOrder[4])
        plt.show()
        fig.savefig("figures/" + slideNames[slide] + "/" + "section" + str(int(section)) + "_IntensityDistributions.png", bbox_inches='tight')
        plt.close(fig) 
    
    # Run GaussianMixture model for each section and channel separatly:

    data = kptn_data_log
    n_samples = np.shape(data)[0]
    dimensions = np.shape(data)[1]
    alpha = np.array(((2,1),(10,1), (10,1), (10,1), (2,1)))

    for section in np.unique(uniqueSectionNumbers):

        for channel in range(5):

            with pm.Model() as model:
                w = pm.Dirichlet('w', alpha[channel])
                mus = pm.Normal('mu', mu = np.array((np.mean(np.sort(data[sectionNumber == section,channel])[:1000])+0.5, np.mean(np.sort(data[sectionNumber == section,channel])[:1000]) + 3)), sigma = np.array((0.33,1)), shape = 2)
                sigmas = pm.Gamma('sigma', mu = np.array((0.1,1)), sigma = np.array((0.2,1)), shape = 2)
                prior = sample_prior(samples = 1000)
                x = pm.NormalMixture('x_obs', w, mus, sigma = sigmas, observed=data[sectionNumber == section,channel])

            # Plot prior for some parameters:
            f = plt.figure()
            plt.hist(prior['mu'])
            plt.show()
            f.savefig("figures/" + slideNames[slide] + "/" + "muPriorSection" + str(int(section)) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 
            
            f = plt.figure()
            plt.hist(prior['sigma'])
            plt.show()
            f.savefig("figures/" + slideNames[slide] + "/" + "sigmaPriorSection" + str(int(section)) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 

            f = plt.figure()
            plt.hist(prior['w'])
            plt.show()
            f.savefig("figures/" + slideNames[slide] + "/" + "wPriorSection" + str(int(section)) + "channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 
            
            # Sample:
            #with model:
            #    %time hmc_trace = pm.sample(draws=300, tune=700, cores=10)

            # Fit:
            with model:
                advi_fit = pm.fit(n=3000, obj_optimizer=pm.adagrad(learning_rate=1e-1), method = 'advi')  

            # Show results MCMC
            #pm.traceplot(hmc_trace)
            #pm.summary(hmc_trace, include_transformed=False)

            # Show results advi:
            f = plt.figure()
            advi_elbo = pd.DataFrame(
                {'log-ELBO': -np.log(advi_fit.hist),
                 'n': np.arange(advi_fit.hist.shape[0])})
            _ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
            f.savefig("figures/" + slideNames[slide] + "/" + "adviElbo_Section" + str(int(section)) + "_channel" + str(channel) + ".png", bbox_inches='tight')
            plt.close(f) 
            advi_trace = advi_fit.sample(10000)
            pm.summary(advi_trace, include_transformed=False)
            
            # Save trace means:
            advi_data = {"advi_mu_0": np.mean(advi_trace.get_values('mu')[:,0]), "advi_mu_1": np.mean(advi_trace.get_values('mu')[:,1]),
                   "advi_sigma_0": np.mean(advi_trace.get_values('sigma')[:,0]), "advi_sigma_1": np.mean(advi_trace.get_values('sigma')[:,1]),
                   "advi_w_0": np.mean(advi_trace.get_values('w')[:,0]), "advi_w_1": np.mean(advi_trace.get_values('w')[:,1])}
            pickle_out = open("data/" + slideNames[slide] + 'Section' + str(int(section)) + "Channel" + str(channel) +'_AdviFitResults.pickle',"wb")
            pickle.dump(advi_data, pickle_out)
            pickle_out.close()

            # Posterior predictive check for MCMC:
            #ppc = pm.sample_posterior_predictive(hmc_trace, samples=100, model=model)

            #fig, ax = plt.subplots(figsize=(8, 6))
            #ax.hist(data[sectionNumber == section,channel], bins=100, density=True,
            #        label='Observed data');
            #ax.hist(ppc['x_obs'].flatten(), bins=100, density=True,
            #        histtype='step', lw=2,
            #        label='Posterior predictive distribution');
            #ax.legend(loc=1);

            # Calculate class membership, by using hmc_trace and logp_normal function:
            #confidenceThreshold = 0.95
            #class0LogProb = [logp_normal(np.mean(hmc_trace.get_values('mu_0')), np.mean(hmc_trace.get_values('sigma')[:,0]) , data[k,channel]) for k in np.where(sectionNumber == section)]
            #class1LogProb = [logp_normal(np.mean(hmc_trace.get_values('mu_1')), np.mean(hmc_trace.get_values('sigma')[:,1]) , data[k,channel]) for k in np.where(sectionNumber == section)]
            #normalizedProbs = [exp_normalize(np.array((class0LogProb[0][i], class1LogProb[0][i]))) for i in range(len(class0LogProb[0]))]
            #maxProbs = [max(normalizedProbs[i]) for i in range(len(normalizedProbs))]
            #classMembership = [np.argmax(normalizedProbs[i]) for i in range(len(normalizedProbs))]
            #confidentClass = [2 if maxProbs[i] < confidenceThreshold else classMembership[i] for i in range(len(classMembership))]

            # Calculate class membership, by using advi_trace and logp_normal function:                            
            confidenceThreshold = 0.95
            class0LogProb = [logp_normal(np.mean(advi_trace.get_values('mu')[:,0]), np.mean(advi_trace.get_values('sigma')[:,0]) , data[k,channel]) for k in np.where(sectionNumber == section)]
            class1LogProb = [logp_normal(np.mean(advi_trace.get_values('mu')[:,1]), np.mean(advi_trace.get_values('sigma')[:,1]) , data[k,channel]) for k in np.where(sectionNumber == section)]
            normalizedProbs = [exp_normalize(np.array((class0LogProb[0][i], class1LogProb[0][i]))) for i in range(len(class0LogProb[0]))]
            maxProbs = [max(normalizedProbs[i]) for i in range(len(normalizedProbs))]
            classMembership = [np.argmax(normalizedProbs[i]) for i in range(len(normalizedProbs))]
            confidentClass = [2 if maxProbs[i] < confidenceThreshold else classMembership[i] for i in range(len(classMembership))]
            # i.e. a value of 2 corresponds to a classification below our confidence threshold, unlike values 0 or 1
            
            # Class membership probability:
            pickle_out = open("data/" + slideNames[slide] + 'Section' + str(int(section)) + "Probability-" + celltypeOrder[channel] + '.pickle',"wb")
            pickle.dump(normalizedProbs, pickle_out)
            pickle_out.close()

            # Histograms:
            if sum(np.array(confidentClass) == 1) > 0:
                boundary1 = min(data[sectionNumber == section,channel][np.array(confidentClass) == 1])
                # i.e. what's the minimum value to confidently classify a cell as positive for this marker?
            else:
                boundary1 = np.inf
            if sum(np.array(confidentClass) == 2) > 0:
                boundary2 = min(data[sectionNumber == section,channel][np.array(confidentClass) == 2])
                # i.e. what's the minimum value to classify a cell into the 'unsure' category
            else:
                boundary2 = 0
            fig = plt.figure()
            fig, ax = plt.subplots()
            N, bins, patches = ax.hist(data[sectionNumber == section,channel], edgecolor='white', linewidth=1, bins = 100)
            for i in range(0, len(patches)):
                if bins[i] < boundary2:
                    patches[i].set_facecolor('blue')   
                elif bins[i] < boundary1:
                    patches[i].set_facecolor('black')
                else:
                    patches[i].set_facecolor('red')
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
            fig.savefig("figures/" + slideNames[slide] + "/" + "ScatterPlotIntensityAndClassification" + "Section" + str(int(section)) + "channel" + channelOrder[channel] + ".png", bbox_inches='tight')  
            plt.close(fig) 
            
            fig = plt.figure()                             
            plt.scatter(kptn_data[sectionNumber == section,0], data[sectionNumber == section,channel], c = colours, s = 0.1)                         
            plt.gca().set_title('Log Intensity and Classification Channel ' + channelOrder[channel])
            plt.show()
            fig.savefig("figures/" + slideNames[slide] + "/" + "ScatterPlotLOGIntensityAndClassification" + "Section" + str(int(section)) + "channel" + channelOrder[channel] + ".png", bbox_inches='tight')
            plt.close(fig) 
            
            # Slide location of each cell type (including unclassified):

            fig = plt.figure()   
            plt.scatter(kptn_data[sectionNumber == section,0][np.array(confidentClass) == 1], kptn_data[sectionNumber == section,1][np.array(confidentClass) == 1], s = 0.05)
            plt.gca().set_title('Nuclei Positive Classification Slide  ' + str(slide) + " Section " + str(int(section)) + " Channel " + channelOrder[channel] + ".png")
            plt.show()
            fig.savefig("figures/" + slideNames[slide] + "/" + "PositiveClassificationPosition" + str(slide) + "section" + str(int(section)) + "channel" + channelOrder[channel] + ".png", bbox_inches='tight')  
            plt.close(fig) 