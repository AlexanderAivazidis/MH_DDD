import numpy as np

def logp_MultiNormal(mu, sigma, value):
    # log probability of individual samples
    k = sigma.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (tt.log(det(sigma)) + (delta(mu).dot(1./sigma) * delta(mu)).sum(axis=1) + k * tt.log(2 * np.pi))

def logp_normal(mu, sigma, value):
    return (-1/2.)*(np.log(2*np.pi) + np.log(sigma**2) + 1/(sigma**2)*(value - mu)**2)