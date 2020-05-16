from reggae.utilities import get_time_square

import tensorflow as tf
from tensorflow import math as tfm

class GPKernelSelector():
    def __init__(self, options, τ):
        self.kernel = options.kernel
        self.N_p = τ.shape[0]
        self.τ = τ
        t_1, t_2 = get_time_square(τ, self.N_p)
        self.t_dist = t_1-t_2
        self.tt = t_1*t_2
        self.t2 = tf.square(t_1)
        self.tprime2 = tf.square(t_2)

    def __call__(self):
        if self.kernel == 'rbf':
            return self.rbf
        elif self.kernel == 'mlp':
            return self.mlp
        else:
            raise Exception('No kernel by that name!')

    def rbf(self, v, l2):
        sq_dist = tf.divide(tfm.square(self.t_dist), tf.reshape(2*l2, (-1, 1, 1)))
        K = tf.reshape(v, (-1, 1, 1)) * tfm.exp(-sq_dist)
        m = tf.zeros((self.N_p), dtype='float64')
        return m, K

    def mlp(self, w, b):
        w = tf.reshape(w, (-1, 1, 1))
        b = tf.reshape(b, (-1, 1, 1))
        denom = tfm.sqrt((w*self.t2 + b + 1) * (w*self.tprime2 + b + 1))
        K = tfm.asin((w*self.tt + b)/denom)
        m = tf.zeros((self.N_p), dtype='float64')
        return m, K

