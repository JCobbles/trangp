import gpflow
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import math as tm
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
        self.N_m = data.m_obs.shape[2]
        Y = data.m_obs[replicate]
        Y_var = data.σ2_m_pre[replicate]
        self.Y = Y.reshape((-1, 1))
        self.Y_var = Y_var.reshape(-1)
        X = np.arange(self.N_m, dtype='float64')*2
        self.X = np.c_[[X for _ in range(self.num_genes)]].reshape(-1, 1)

        self.kernel = LinearResponseKernel(data, self.Y_var)
        self.meanfunc_exp = LinearResponseMeanFunction(data, self.kernel)
        self.internal_model = gpflow.models.GPR(
            data=(self.X, self.Y), 
            kernel=self.kernel, 
            mean_function=self.meanfunc_exp
        )

    def objective_closure(self):
        ret = - self.internal_model.log_marginal_likelihood()
        return ret

    def fit(self, maxiter=50):
        opt = gpflow.optimizers.Scipy()
        start = timer()
        opt_logs = opt.minimize(self.objective_closure,
                                self.internal_model.trainable_variables,
                                options=dict(maxiter=maxiter, disp=True, eps=0.00000001), 
                                method='CG') # CG: 27.0
        end = timer()
        print(f'Time taken: {(end - start):.04f}s')
        return opt_logs

    def predict_x(self, pred_t):
        K_xx = self.kernel.K(self.X, None)
        K_inv = tf.linalg.inv(K_xx)
        K_xstarx = tf.transpose(self.kernel.K_xstarx(self.X[:self.N_m], pred_t))
        KxstarxKinvY = tf.linalg.matvec(tf.matmul(K_xstarx, K_inv), tf.reshape(self.Y, -1))
        return tf.reshape(KxstarxKinvY, (self.num_genes, pred_t.shape[0]))

    def predict_f(self, pred_t):
        Kxx = self.kernel.K(self.X, None)
        K_inv = tf.linalg.inv(Kxx)
        Kxf = self.kernel.K_xf(self.X, pred_t)
        KfxKxx = tf.matmul(tf.transpose(Kxf), K_inv)
        return tf.matmul(KfxKxx, self.Y)
