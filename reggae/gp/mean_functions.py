import gpflow
from gpflow.utilities import positive

import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from reggae.utilities import broadcast_tile


class LinearResponseMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, data, k_exp):
        initial_B = tf.reduce_mean(data.m_obs[0], axis=1)*k_exp.D
        self.B = gpflow.Parameter(B=initial_B, transform=positive())

    def __call__(self, X):
        ret = tf.zeros(0, dtype='double')
        block_size = int(X.shape[0]/num_genes)
        
        ret = broadcast_tile(tf.reshape(self.B / k_exp.D, (-1, 1)), block_size, 1)

        return ret

