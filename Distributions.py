import numpy as np
from scipy import stats


class GaussianDistribution:
    def __init__(self, mu=None, std_dev=None):
        self.mu = mu if mu is not None else 4
        self.std_dev = std_dev if std_dev is not None else 1
        self.type = stats.norm
        self.rng = self.type(loc=self.mu, scale=self.std_dev)

    def sample(self, n):
        return np.sort(self.rng.rvs(n))

    def get_rng(self):
        return self.rng

    def get_type(self):
        return self.type

    def get_loc(self):
        return self.mu

    def get_scale(self):
        return self.std_dev


class UniformDistribution:
    def __init__(self, low_bound=None, high_bound=None, noise_level=None):
        self.low_bound = low_bound if low_bound is not None else -8
        self.high_bound = high_bound if high_bound is not None else 8
        self.noise_level = noise_level if noise_level is not None else 0.01
        self.type = stats.uniform
        self.rng = self.type(loc=self.low_bound, scale=abs(self.high_bound - self.low_bound))

    def _sample_linspace(self, n):
        lin_range = np.linspace(self.low_bound, self.high_bound, n)
        lin_range += np.random.random(n) * self.noise_level
        return lin_range

    def _sample_uniform(self, n):
        return np.sort(self.rng.rvs(n))

    def sample(self, n):
        return self._sample_uniform(n)

    def get_rng(self):
        return self.rng

    def get_loc(self):
        return self.low_bound

    def get_scale(self):
        return abs(self.high_bound - self.low_bound) / 2.0  # by the computation of (loc, loc+scale) in scipy.stats

    def get_type(self):
        return self.type


class Distribution:
    """
    Distribution object that holds a some type of distribution and can sample from it arbitrary long sequences (samples)
    """
    def __init__(self, dist_type="gaussian", **kwargs):
        if dist_type == "gaussian":
            self.distributor = GaussianDistribution(mu=kwargs.get("mu"), std_dev=kwargs.get("std_dev"))
        elif dist_type == "uniform":
            self.distributor = UniformDistribution(low_bound=kwargs.get("low_bound"),
                                                   high_bound=kwargs.get("high_bound"),
                                                   noise_level=kwargs.get("noise_level"))
        else:
            raise NotImplementedError("Type given: {type}, is non implemented".format(type=dist_type))

    def sample(self, n):
        return self.distributor.sample(n)

    def get_rng(self):
        return self.distributor.get_rng()

    def get_type(self):
        return self.distributor.get_type()

    def get_loc(self):
        return self.distributor.get_loc()

    def get_scale(self):
        return self.distributor.get_scale()

#
# #
# ##
# class UniformDistribution:
#     def __init__(self, low_bound=None, high_bound=None, noise_level=None):
#         self.low_bound = low_bound if low_bound is not None else -8
#         self.high_bound = high_bound if high_bound is not None else 8
#         self.noise_level = noise_level if noise_level is not None else 0.01
#
#     def _sample_linspace(self, n):
#         lin_range = np.linspace(self.low_bound, self.high_bound, n)
#         lin_range += np.random.random(n) * self.noise_level
#         return lin_range
#
#     def _sample_uniform(self, n):
#         return np.sort(np.random.uniform(low=self.low_bound, high=self.high_bound, size=n))
#
#     def sample(self, n):
#         return self._sample_linspace(n)

