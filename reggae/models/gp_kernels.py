from reggae.utilities import get_time_square

import tensorflow as tf
from tensorflow import math as tfm
import numpy as np

f64 = np.float64

class GPKernelSelector():
    def __init__(self, data, options):
        self.kernel = options.kernel
        self.τ = data.τ        
        self.N_p = data.τ.shape[0]
        self.num_tfs = data.f_obs.shape[1]
        t_1, t_2 = get_time_square(self.τ, self.N_p)
        self.t_dist = t_1-t_2
        self.tt = t_1*t_2
        self.t2 = tf.square(t_1)
        self.tprime2 = tf.square(t_2)

        min_dist = min(data.t[1:]-data.t[:-1])
        self._ranges = {
            'rbf': [(f64(1e-4), f64(1+max(np.var(data.f_obs, axis=2)))),
                    (f64(min_dist**2)-0.2, f64(data.t[-1]**2))],
            'mlp': [(f64(1), f64(10)), (f64(3.5), f64(20))],
        }

    def __call__(self):
        if self.kernel == 'rbf':
            return self.rbf
        elif self.kernel == 'mlp':
            return self.mlp
        else:
            raise Exception('No kernel by that name!')

    def initial_params(self):
        if self.kernel == 'rbf':
            return [0.8*tf.ones(self.num_tfs, dtype='float64'), 
                    0.98*tf.ones(self.num_tfs, dtype='float64')]
        elif self.kernel == 'mlp':
            return [0.8*tf.ones(self.num_tfs, dtype='float64'), 
                    0.98*tf.ones(self.num_tfs, dtype='float64')]
    def ranges(self):
        return self._ranges[self.kernel]

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

