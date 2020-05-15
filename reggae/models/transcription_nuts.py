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
from reggae.utilities import rotate, get_rbf_dist, jitter_cholesky, logit, logistic, LogisticNormal

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
        w = logit(w)
        w_0 = logit(w_0)
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
        # w = 1*tf.ones((self.num_genes, self.num_tfs), dtype='float64') # TODO
        # w_0 = tf.zeros(self.num_genes, dtype='float64') # TODO
        w = all_states[state_indices['weights']][0] if w is None else w
        w_0 = all_states[state_indices['weights']][1] if w_0 is None else w_0
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
        variance = logit(tf.reshape(σ2_m, (-1, 1)))
        if self.preprocessing_variance:
            variance = variance + self.data.σ2_m_pre # add PUMA variance
#         print(variance.shape, sq_diff.shape)
        log_lik = -0.5*tfm.log(2*np.pi*(variance)) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik, axis=1)
        return log_lik, sq_diff

    def tfs(self, σ2_f, fbar, return_sq_diff=False): 
        '''
        Computes log-likelihood of the transcription factors.
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


TupleParams_pre = collections.namedtuple('TupleParams_pre', [
    'fbar','k_fbar','kbar','σ2_m','weights','L','V', 'Δ', 'kinetics', 'σ2_f',
])
TupleParams = collections.namedtuple('TupleParams', [
    'fbar','k_fbar','kbar','σ2_m','weights','L','V', 'Δ', 'kinetics',
])

class TranscriptionMixedSampler():
    '''
    Data is a tuple (m, f) of shapes (num, time)
    time is a tuple (t, τ, common_indices)
    '''
    def __init__(self, data: DataHolder, options: Options):
        self.data = data
        self.samples = None
        min_dist = min(data.t[1:]-data.t[:-1])
        self.N_p = data.τ.shape[0]
        self.N_m = data.t.shape[0]      # Number of observations

        self.num_tfs = data.f_obs.shape[0] # Number of TFs
        self.num_genes = data.m_obs.shape[0]

        self.likelihood = TranscriptionLikelihood(data, options)
        self.options = options

        self.state_indices = {
            'kinetics': 0,
            'fbar': 1, 
            'rbf_params': 2,
            'σ2_m': 3,
            'Δ': 4,
            'weights': 5,
            'σ2_f': 6,
        }
        logistic_step_size = 0.00001

        # Interaction weights
        def w_log_prob(all_states):
            def w_log_prob_fn(wstar, w_0star):
                new_prob = tf.reduce_sum(self.likelihood.genes(
                    all_states=all_states, 
                    state_indices=self.state_indices,
                    w=wstar,
                    w_0=w_0star))
                new_prob += tf.reduce_sum(self.params.weights.prior[0].log_prob(logit(wstar))) 
                new_prob += tf.reduce_sum(self.params.weights.prior[1].log_prob(logit(w_0star)))
                return tf.reduce_sum(new_prob)
            return w_log_prob_fn
        w = Parameter('w', LogisticNormal(f64(-1), f64(1)), logistic(1*tf.ones((self.num_genes, self.num_tfs), dtype='float64')))
        w_0 = Parameter('w_0', LogisticNormal(f64(-1), f64(1)), 0.5*tf.ones(self.num_genes, dtype='float64'))
        weights = Parameter('weights', [w.prior, w_0.prior], [w.value, w_0.value], step_size=logistic_step_size,
                            hmc_log_prob=w_log_prob, requires_all_states=True)
        # Latent function
        def fbar_log_prob(all_states):
            def fbar_log_prob_fn(fstar):
                vbar, l2bar = all_states[self.state_indices['rbf_params']]
                
                new_m_likelihood = self.likelihood.genes(
                    all_states,
                    self.state_indices,
                    fbar=fstar,
                )
                σ2_f = all_states[self.state_indices['σ2_f']]
                new_f_likelihood = tf.cond(tf.equal(self.options.tf_mrna_present, tf.constant(True)),
                                           lambda:tf.reduce_sum(self.likelihood.tfs(
                                               σ2_f,
                                               fstar
                                           )), lambda:f64(0))
                new_f_prior = tf.reduce_sum(self.fbar_prior(fstar, vbar, l2bar))
                new_prob = tf.reduce_sum(new_m_likelihood) + new_f_likelihood + new_f_prior
                return new_prob
            return fbar_log_prob_fn

        fbar_kernel = FKernel(self.likelihood, 
                              self.fbar_prior_params, 
                              self.num_tfs, self.num_genes, 
                              self.options.tf_mrna_present, 
                              self.state_indices,
                              0.005*tf.ones(self.N_p, dtype='float64'))
        fbar_initial = 0.5*tf.ones((self.num_tfs, self.N_p), dtype='float64')
        if self.options.latent_function_metropolis:
            fbar = Parameter('fbar', self.fbar_prior, fbar_initial,
                             kernel=fbar_kernel, requires_all_states=False)
        else:
            fbar = Parameter('fbar', self.fbar_prior, fbar_initial, hmc_log_prob=fbar_log_prob,
                             requires_all_states=True, step_size=logistic_step_size)

        # GP hyperparameters
        def rbf_params_log_prob(all_states):
            def rbf_params_log_prob(vbar, l2bar):
                v = logit(vbar, nan_replace=self.params.V.prior.b)
                l2 = logit(l2bar, nan_replace=self.params.L.prior.b)

                new_prob = tf.reduce_sum(self.params.fbar.prior(all_states[self.state_indices['fbar']], vbar, l2bar))
                new_prob += self.params.V.prior.log_prob(v)
                new_prob += self.params.L.prior.log_prob(l2)
#                 tf.print('new prob', new_prob)
#                 if new_prob < -1e3:
#                     tf.print(all_states[self.state_indices['fbar']], v, l2)
                return tf.reduce_sum(new_prob)
            return rbf_params_log_prob

        V = Parameter('rbf_params', LogisticNormal(f64(1e-4), f64(1+max(np.var(data.f_obs, axis=1))),allow_nan_stats=False), 
                      [0.76*tf.ones(self.num_tfs, dtype='float64'), 0.94*tf.ones(self.num_tfs, dtype='float64')], 
                      step_size=logistic_step_size, fixed=not options.tf_mrna_present, 
                      hmc_log_prob=rbf_params_log_prob, requires_all_states=True)
        L = Parameter('L', LogisticNormal(f64(min_dist**2), f64(data.t[-1]**2), allow_nan_stats=False), None)

        self.t_dist = get_rbf_dist(data.τ, self.N_p)

        # Translation kinetic parameters
        k_fbar = Parameter('k_fbar', LogisticNormal(0.1, 3), 0.7*tf.ones((self.num_tfs,2), dtype='float64'))

        # White noise for genes
        # if not options.preprocessing_variance:
        #     σ2_f_kernel = GibbsKernel(self.options)

        def σ2_m_log_prob(all_states):
            def σ2_m_log_prob_fn(σ2_mstar):
                # tf.print('starr:', logit(σ2_mstar))
                new_prob = self.likelihood.genes(
                    all_states=all_states, 
                    state_indices=self.state_indices,
                    σ2_m=σ2_mstar 
                ) + self.params.σ2_m.prior.log_prob(logit(σ2_mstar))
                # tf.print('prob', tf.reduce_sum(new_prob))
                return tf.reduce_sum(new_prob)                
            return σ2_m_log_prob_fn
        σ2_m = Parameter('σ2_m', LogisticNormal(f64(1e-5), f64(max(np.var(data.f_obs, axis=1)))), 
                         logistic(f64(5e-3))*tf.ones(self.num_genes, dtype='float64'), 
                         hmc_log_prob=σ2_m_log_prob, requires_all_states=True, step_size=logistic_step_size)
        # Transcription kinetic parameters
        def constrain_kbar(kbar, gene):
            '''Constrains a given row in kbar'''
#             if gene == 3:
#                 kbar[2] = np.log(0.8)
#                 kbar[3] = np.log(1.0)
            kbar[kbar < -10] = -10
            kbar[kbar > 3] = 3
            return kbar
        kbar_initial = 0.6*np.float64(np.c_[ # was -0.1
            np.ones(self.num_genes), # a_j
            np.ones(self.num_genes), # b_j
            np.ones(self.num_genes), # d_j
            np.ones(self.num_genes)  # s_j
        ])
        def kbar_log_prob(all_states):
            def kbar_log_prob_fn(kbar, k_fbar):
                k_m = logit(kbar)
                k_f = logit(k_fbar)
                new_prob = self.likelihood.genes(
                    all_states=all_states,
                    state_indices=self.state_indices,
                    kbar=kbar,
                    k_fbar=k_fbar,
                ) 
                new_prob += tf.reduce_sum(self.params.kbar.prior.log_prob(k_m))
                new_prob += tf.reduce_sum(self.params.k_fbar.prior.log_prob(k_f))
                return tf.reduce_sum(new_prob)
            return kbar_log_prob_fn
        for j, k in enumerate(kbar_initial):
            kbar_initial[j] = constrain_kbar(k, j)
        kbar = Parameter('kbar', LogisticNormal(0.01, 8), kbar_initial, constraint=constrain_kbar)
        
        kinetics = Parameter('kinetics', None, [kbar.value, k_fbar.value],
                             hmc_log_prob=kbar_log_prob, constraint=constrain_kbar, 
                             step_size=logistic_step_size, requires_all_states=True)


        delta_kernel = DeltaKernel(self.likelihood, 0, 10, self.state_indices, tfd.Exponential(f64(0.3)))
        Δ = Parameter('Δ', tfd.InverseGamma(f64(0.01), f64(0.01)), 0.6*tf.ones(self.num_tfs, dtype='float64'),
                        kernel=delta_kernel, requires_all_states=False)
        
        if not options.preprocessing_variance:
            kernel = GibbsKernel(data, options, self.likelihood, tfd.InverseGamma(f64(0.01), f64(0.01)), self.state_indices)
            σ2_f = Parameter('σ2_f', None, 1e-4*tf.ones((self.num_tfs,1), dtype='float64'), kernel=kernel)
            self.params = TupleParams_pre(fbar, k_fbar, kbar, σ2_m, weights, L, V, Δ, kinetics, σ2_f)
        else:
            self.params = TupleParams(fbar, k_fbar, kbar, σ2_m, weights, L, V, Δ, kinetics)
            
    def fbar_prior_params(self, vbar, l2bar):
        v = logit(vbar, nan_replace=self.params.V.prior.b)
        l2 = logit(l2bar, nan_replace=self.params.L.prior.b)

#         tf.print('vl2', v, l2)
        sq_dist = tf.divide(tfm.square(self.t_dist), tf.reshape(2*l2, (-1, 1, 1)))
        K = tf.reshape(v, (-1, 1, 1)) * tfm.exp(-sq_dist)
        m = tf.zeros((self.N_p), dtype='float64')
        return m, K

    def fbar_prior(self, fbar, v, l2):
        m, K = self.fbar_prior_params(v, l2)
#         tf.print(fbar[0][:6])
        jitter = tf.linalg.diag(1e-8 *tf.ones(self.N_p, dtype='float64'))
        prob = 0
        for i in range(self.num_tfs):
            prob += tfd.MultivariateNormalTriL(loc=m, scale_tril=tf.linalg.cholesky(K[i]+jitter)).log_prob(fbar[i])
        return prob


    def sample(self, T=2000, store_every=10, burn_in=1000, report_every=100, num_chains=4):
        print('----- Sampling Begins -----')
        
        params = self.params
        progbar = tf.keras.utils.Progbar(
            100, width=30, verbose=1, interval=0.05, stateful_metrics=None,
            unit_name='step'
        )

        active_params = [
            params.kinetics,
            params.fbar,
            params.V,
            params.σ2_m
        ]
        if self.options.delays:
            active_params += [params.Δ]
        active_params += [
            params.weights,
        ]
        if not self.options.preprocessing_variance:
            active_params += [params.σ2_f]

        kernels = [param.kernel for param in active_params]
#         if self.options.tf_mrna_present:
        send_all_states = [param.requires_all_states for param in active_params]

        current_state = [param.value for param in active_params]
        mixed_kern = MixedKernel(kernels, send_all_states)
        
        def trace_fn(a, previous_kernel_results):
            return previous_kernel_results.is_accepted

        # Run the chain (with burn-in).
        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
                  num_results=T,
                  num_burnin_steps=burn_in,
                  current_state=current_state,
                  kernel=mixed_kern,
                  trace_fn=trace_fn)

            return samples, is_accepted

        samples, is_accepted = run_chain()

        add_to_previous = (self.samples is not None)
        for param in active_params:
            index = self.state_indices[param.name]
            param_samples = samples[index]
            if type(param_samples) is list:
                if add_to_previous:
                    for i in range(len(param_samples)):
                        self.samples[index][i] = tf.concat([self.samples[index][i], samples[index][i]], axis=0)
                param_samples = [[param_samples[i][-1] for i in range(len(param_samples))]]
            else:
                if add_to_previous:
                    self.samples[index] = tf.concat([self.samples[index], samples[index]], axis=0)
            param.value = param_samples[-1]

        if not add_to_previous:
            self.samples = samples     
        self.is_accepted = is_accepted
        print('----- Finished -----')
        return samples, is_accepted
        
            
    @staticmethod
    def initialise_from_state(args, state):
        model = TranscriptionMixedSampler(*args)
        model.acceptance_rates = state.acceptance_rates
        model.samples = state.samples
        return model

    def predict_m(self, kbar, k_fbar, w, fbar, w_0):
        return self.likelihood.predict_m(kbar, k_fbar, w, fbar, w_0)

    def predict_m_with_current(self):
        return self.likelihood.predict_m(self.params.kbar.value, 
                                         self.params.k_fbar.value, 
                                         self.params.weights.value[0], 
                                         self.params.fbar.value,
                                         self.params.weights.value[1])
