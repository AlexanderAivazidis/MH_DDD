import numpy as np
from scipy import stats
from theano.tensor.nlinalg import det, matrix_inverse, trace, eigh
from pymc3.distributions import transforms
from pymc3.distributions.distribution import Continuous, Discrete, draw_values, generate_samples
from pymc3.distributions.special import gammaln, multigammaln
from pymc3.distributions.dist_math import bound, logpow, factln
import theano.tensor as T

def logp_MultiNormal(mu, sigma, value):
    # log probability of individual samples
    k = sigma.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (tt.log(det(sigma)) + (delta(mu).dot(1./sigma) * delta(mu)).sum(axis=1) + k * tt.log(2 * np.pi))

def logp_normal(mu, sigma, value):
    return (-1/2.)*(np.log(2*np.pi) + np.log(sigma**2) + 1/(sigma**2)*(value - mu)**2)

class MvNormal(Continuous):
    r"""
    This MvNormal can handle tensor arguments (to some degree).  Also, the sampling routine uses the correct covariance,
    but it's costly.
    """
    def __init__(self, mu, tau, *args, **kwargs):
        super(MvNormal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau = tau

    def random(self, point=None, size=None):
        mu, tau = draw_values([self.mu, self.tau], point=point)

        def _random(mean, tau, size=None):
            supp_dim = mean.shape[-1]
            mus_collapsed = mean.reshape((-1, supp_dim))
            taus_collapsed = tau.reshape((-1, supp_dim, supp_dim))
            # FIXME: do something smarter about tau/cov
            covs_collapsed = np.apply_over_axes(lambda x,y: np.linalg.inv(x), taus_collapsed, 0)

            from functools import partial
            mvrvs = partial(stats.multivariate_normal.rvs, size=1)
            res = map(mvrvs, mus_collapsed, covs_collapsed)

            # FIXME: this is a hack; the PyMC sampling framework
            # will incorrectly set `size == Distribution.shape` when a single
            # sample is requested, implying that we want
            # `Distribution.shape`-many samples of a
            # `Distribution.shape` sized object: too many!  That's why
            # we're ignoring `size` right now and only ever asking
            # for a single sample.
            return np.asarray(res).reshape(mean.shape)

        samples = generate_samples(_random,
                                   mean=mu, tau=tau,
                                   dist_shape=self.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        mu = T.as_tensor_variable(self.mu)
        tau = T.as_tensor_variable(self.tau)

        reps_shape_T = tau.shape[:-2]
        reps_shape_prod = T.prod(reps_shape_T, keepdims=True)
        dist_shape_T = mu.shape[-1:]

        # collapse reps dimensions
        flat_supp_shape = T.concatenate((reps_shape_prod, dist_shape_T))
        mus_collapsed = mu.reshape(flat_supp_shape, ndim=2)
        taus_collapsed = tau.reshape(T.concatenate((reps_shape_prod,
            dist_shape_T, dist_shape_T)), ndim=3)

        # force value to conform to reps_shape
        value_reshape = T.ones_like(mu) * value
        values_collapsed = value_reshape.reshape(flat_supp_shape, ndim=2)

        def single_logl(_mu, _tau, _value, k):
            delta = _value - _mu
            result = k * T.log(2 * np.pi) + T.log(det(_tau))
            result += T.square(delta.dot(_tau)).sum(axis=-1)
            return -result/2

        from theano import scan
        res, _ = scan(fn=single_logl
                , sequences=[mus_collapsed, taus_collapsed, values_collapsed]
                , non_sequences=[dist_shape_T]
                , strict=True
                )
        return res.sum()