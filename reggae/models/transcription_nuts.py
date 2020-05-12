import collections

import tensorflow as tf
from tensorflow import math as tfm
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from reggae.mcmc import MetropolisHastings, Parameter, MetropolisKernel
from reggae.models.results import GenericResults
from reggae.models import Options
from reggae.models.kernels import FKernel, MixedKernel, DeltaKernel, GibbsKernel
from reggae.data_loaders import DataHolder
from reggae.utilities import rotate, get_rbf_dist, jitter_cholesky, logit, LogisticNormal

import numpy as np
from scipy.special import expit

f64 = np.float64

class TranscriptionLikelihood():
    def __init__(self, data: DataHolder, options: Options):
        self.options = options
        self.data = data
        self.preprocessing_variance = options.preprocessing_variance
        self.num_genes = data.m_obs.shape[0]
        self.num_tfs = data.f_obs.shape[0]

    def calculate_protein(self, fbar, k_fbar, Δ=None): # Calculate p_i vector
        τ = self.data.τ
        f_i = tfm.log(1+tfm.exp(fbar))
        a_i, δ_i = (tf.reshape(logit(k_fbar[:, i]), (-1, 1)) for i in range(2))
        if self.options.delays:
            # Add delay
            Δ = tf.cast(Δ, 'int32')
            # tf.print('adding delay', Δ)
            f_i = rotate(f_i, -Δ)
            mask = ~tf.sequence_mask(Δ, f_i.shape[1])
            f_i = tf.where(mask, f_i, 0)
            # print(f_i)

        # Approximate integral (trapezoid rule)
        resolution = τ[1]-τ[0]
        sum_term = tfm.multiply(tfm.exp(δ_i*τ), f_i)
        integrals = tf.concat([tf.zeros((self.num_tfs, 1), dtype='float64'), 
                         0.5*resolution*tfm.cumsum(sum_term[:, :-1] + sum_term[:, 1:], axis=1)], axis=1) 
        exp_δt = tfm.exp(-δ_i*τ)
        p_i = a_i * exp_δt + exp_δt * integrals
        return p_i

    @tf.function
    def predict_m(self, kbar, k_fbar, w, fbar, w_0, Δ=None):
        # Take relevant parameters out of log-space
        a_j, b_j, d_j, s_j = (tf.reshape(logit(kbar[:, i]), (-1, 1)) for i in range(4))
        τ = self.data.τ
        N_p = self.data.τ.shape[0]

        p_i = self.calculate_protein(fbar, k_fbar, Δ)
        # Calculate m_pred
        resolution = τ[1]-τ[0]
        integrals = tf.zeros((self.num_genes, N_p))
        interactions =  tf.matmul(w, tfm.log(p_i+1e-100)) + w_0[:, None]
        G = tfm.sigmoid(interactions) # TF Activation Function (sigmoid)
        sum_term = G * tfm.exp(d_j*τ)
        integrals = tf.concat([tf.zeros((self.num_genes, 1), dtype='float64'), # Trapezoid rule
                               0.5*resolution*tfm.cumsum(sum_term[:, :-1] + sum_term[:, 1:], axis=1)], axis=1) 

        exp_dt = tfm.exp(-d_j*τ)
        integrals = tfm.multiply(exp_dt, integrals)
        m_pred = b_j/d_j + tfm.multiply((a_j-b_j/d_j), exp_dt) + s_j*integrals

        return m_pred

    def genes(self, all_states=None, state_indices=None,
              k_fbar=None,
              fbar=None, 
              kbar=None, 
              w=None,
              w_0=None,
              σ2_m=None, 
              Δ=None, return_sq_diff=False):
        '''
        Computes likelihood of the genes.
        If any of the optional args are None, they are replaced by their 
        current value in all_states.
        '''

        k_fbar = all_states[state_indices['kinetics']][1] if k_fbar is None else k_fbar
        fbar = all_states[state_indices['fbar']] if fbar is None else fbar
        kbar = all_states[state_indices['kinetics']][0] if kbar is None else kbar
        w = 1*tf.ones((self.num_genes, self.num_tfs), dtype='float64') # TODO
        w_0 = tf.zeros(self.num_genes, dtype='float64') # TODO
        # w = all_states[state_indices['w']][0] if w is None else w
        # w_0 = all_states[state_indices['w']][1] if w_0 is None else w_0
        σ2_m = all_states[state_indices['σ2_m']] if σ2_m is None else σ2_m
        if self.options.delays:
            Δ = all_states[state_indices['Δ']] if Δ is None else Δ
        lik, sq_diff = self._genes(k_fbar, fbar, kbar, w, w_0, σ2_m, Δ)

        if return_sq_diff:
            return lik, sq_diff
        return lik

    @tf.function
    def _genes(self, k_fbar, fbar, kbar, w, w_0, σ2_m, Δ=None):
        m_pred = self.predict_m(kbar, k_fbar, w, fbar, w_0, Δ)

        sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred),self.data.common_indices)))
        variance = tf.reshape(σ2_m, (-1, 1))
        if self.preprocessing_variance:
            variance = variance + self.data.σ2_m_pre # add PUMA variance
#         print(variance.shape, sq_diff.shape)
        log_lik = -0.5*tfm.log(2*np.pi*(variance)) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik, axis=1)
        return log_lik, sq_diff

    def tfs(self, σ2_f, fbar, return_sq_diff=False): 
        '''
        Computes log-likelihood of the transcription factors.
        TODO this should be for the i-th TF
        '''
        assert self.options.tf_mrna_present
        if not self.preprocessing_variance:
            variance = tf.reshape(σ2_f, (-1, 1))
        else:
            variance = self.data.σ2_f_pre
        f_pred = tfm.log(1+tfm.exp(fbar))
        sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))

        log_lik = -0.5*tfm.log(2*np.pi*variance) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik, axis=1)
        if return_sq_diff:
            return log_lik, sq_diff
        return log_lik
