import numpy as np
import pandas as pd
from scipy.special import factorial


# UTILITIES FOR COMPUTING FACTORIAL

def stirling_approx(n):
    return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)


def log_fact(n):  # use Stirling approx
    # the approximation actually holds for large enough n, and for small values you have no overflow
    # n is actually a vector. 
    fact = lambda t: np.log(factorial(t)) if t < 21 else stirling_approx(t)
    fcn = np.vectorize(fact)
    return fcn(n)


# MODEL CLASSES. ONLY POISSON (WITH EXPONENTIAL PRIOR) AND GEOMETRIC (WITH UNIFORM PRIOR) ARE IMPLEMENTED UP TO NOW:

class GenericModel():

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def set_obs(self, obs):
        self.obs = obs
        self.s_1 = self.s()
        self.t_1 = self.t()

    def s(self):
        return np.sum(self.obs)

    def t(self):
        return np.sum(log_fact(self.obs))

    def gen_sample(self):
        self.gen_param()

        self.obs = self.sample_distribution(self.param, self.n_samples)
        self.s_1 = self.s()
        self.t_1 = self.t()

        return pd.Series(data=(self.param, self.s_1, self.t_1), index=("param", "s_1", "t_1"))


class PoissonModel(GenericModel):

    # in the present implementation, each call to the gen_prior function

    def __init__(self, n_samples):
        super(PoissonModel, self).__init__(n_samples)
        self.sample_distribution = np.random.poisson

    def gen_param(self):  # we place an exponential (1) prior on the Poisson parameter
        self.param = np.random.exponential()

    def evidence(self):  # computed in logscale
        return log_fact(self.s_1) - self.t_1 - (self.s_1 + 1) * np.log(self.n_samples + 1)


class GeometricModel(GenericModel):

    # in the present implementation, each call to the gen_prior function

    def __init__(self, n_samples):
        super(GeometricModel, self).__init__(n_samples)
        self.sample_distribution = lambda p, n_samples: np.random.geometric(p, n_samples) - 1

    def gen_param(self):  # we place a Uniform prior on the geometric parameter
        self.param = np.random.uniform()

    def evidence(self):
        return log_fact(self.s_1) + log_fact(self.n_samples) - log_fact(self.n_samples + self.s_1 + 1)
