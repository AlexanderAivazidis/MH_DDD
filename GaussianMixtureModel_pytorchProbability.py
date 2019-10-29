import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline
sns.set_context('paper')
sns.set_style('darkgrid')
import os
os.chdir('/home/jovyan/MH_DDD/')
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import pymc3 as pm, theano.tensor as tt
from pymc3.math import logsumexp

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

#### Run GaussianMixture model:

x = kptn_data_log[1:10000,3:4]
n_comp = 2
concentration = 0.001

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, tau):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau, value)
                 for i, mu in enumerate(mus)]

        return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

with pm.Model() as model:
    mus = [MvNormal('mu_%d' % i,
                    mu=pm.floatX(np.zeros(2)),
                    tau=pm.floatX(0.1 * np.eye(2)),
                    shape=(2,))
           for i in range(2)]
    pi = Dirichlet('pi', a=pm.floatX(0.1 * np.ones(2)), shape=(2,))
    xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data)

with model:
    fit_advi = pm.fit(method='advi', n = 60000)
    
advi_elbo = pd.DataFrame(
    {'log-ELBO': -np.log(fit_advi.hist),
     'n': np.arange(fit_advi.hist.shape[0])})

_ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)

advi_trace = fit_advi.sample(10000)
pm.traceplot(advi_trace)
pm.summary(advi_trace)

# display class predictions by the model in 2D:
CB = plt.colorbar(CS, shrink=0.8, extend='both')
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

# Display intensity distributions for each component:
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