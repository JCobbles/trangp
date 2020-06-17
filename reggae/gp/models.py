import gpflow
import numpy as np

import tensorflow as tf
from tensorflow import math as tm
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability import mcmc
import tensorflow_probability as tfp

from reggae.gp.options import Options
from reggae.data_loaders import DataHolder
from reggae.gp import LinearResponseKernel, LinearResponseMeanFunction

'''Analytical Linear Response
'''
class LinearResponseModel():

    def __init__(self, data: DataHolder, options: Options, replicate=0):
        self.data = data
        self.num_genes = data.m_obs.shape[1]
        N_m = data.m_obs.shape[2]
        Y = data.m_obs[replicate]
        Y_var = data.σ2_m_pre[replicate]
        self.Y = Y.reshape((-1, 1))
        self.Y_var = Y_var.reshape(-1)
        X = np.arange(N_m, dtype='float64')*2
        self.X = np.c_[[X for _ in range(self.num_genes)]].reshape(-1, 1)

        self.k_exp = LinearResponseKernel(self.num_genes)
        self.meanfunc_exp = LinearResponseMeanFunction(data, self.k_exp)
        self.internal_model = gpflow.models.GPR(
            data=(X, Y), 
            kernel=self.k_exp, 
            mean_function=self.meanfunc_exp
        )

    def predict_x(self):
        pass

    def predict_f(self):
        pass