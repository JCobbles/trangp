import collections

import tensorflow as tf
from tensorflow import math as tfm
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from reggae.mcmc import MetropolisHastings
from reggae.mcmc.parameter import KernelParameter, Parameter
from reggae.models.results import GenericResults, SampleResults
from reggae.models import Options, GPKernelSelector
from reggae.mcmc.kernels import FKernel, MixedKernel, DeltaKernel, GibbsKernel
from reggae.data_loaders import DataHolder
from reggae.utilities import rotate, jitter_cholesky, logit, logistic, LogisticNormal, inverse_positivity

import numpy as np
from scipy.special import expit

f64 = np.float64

class TranscriptionLikelihood():
    def __init__(self, data: DataHolder, options: Options):
        self.options = options
        self.data = data
        self.preprocessing_variance = options.preprocessing_variance
        self.num_genes = data.m_obs.shape[1]
        self.num_tfs = data.f_obs.shape[1]
        self.num_replicates = data.f_obs.shape[0]

    def calculate_protein(self, fbar, k_fbar, Δ=None): # Calculate p_i vector
        τ = self.data.τ
        f_i = inverse_positivity(fbar)
        a_i, δ_i = (tf.reshape(logit(k_fbar[:, i]), (-1, 1)) for i in range(2))
        if self.options.delays:
            # Add delay 
            Δ = tf.cast(Δ, 'int32')

            for r in range(self.num_replicates):
                f_ir = rotate(f_i[r], -Δ)
                mask = ~tf.sequence_mask(Δ, f_i.shape[2])
                f_ir = tf.where(mask, f_ir, 0)
                mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
                mask[r] = 1
                f_i = (1-mask) * f_i + mask * f_ir

        p_i = tf.zeros_like(f_i)
        for r in range(self.num_replicates):
            # Approximate integral (trapezoid rule)
            resolution = τ[1]-τ[0]
            sum_term = tfm.multiply(tfm.exp(δ_i*τ), f_i[r])
            integrals = tf.concat([tf.zeros((self.num_tfs, 1), dtype='float64'), 
                                   0.5*resolution*tfm.cumsum(sum_term[:, :-1] + sum_term[:, 1:], axis=1)], axis=1) 
            exp_δt = tfm.exp(-δ_i*τ)
            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            p_ir = a_i * exp_δt + exp_δt * integrals
            p_i = (1-mask)*p_i + mask * p_ir
        return p_i

    @tf.function
    def predict_m(self, kbar, k_fbar, wbar, fbar, w_0bar, Δ=None):
        # Take relevant parameters out of log-space
        a_j, b_j, d_j, s_j = (tf.reshape(logit(kbar[:, i]), (-1, 1)) for i in range(4))
        w = logit(wbar)
        w_0 = logit(w_0bar)
        τ = self.data.τ
        N_p = self.data.τ.shape[0]

        p_i = self.calculate_protein(fbar, k_fbar, Δ)

        # Calculate m_pred
        m_pred = tf.zeros((self.num_replicates, self.num_genes, N_p), dtype='float64')
        for r in range(self.num_replicates):
            resolution = τ[1]-τ[0]
            integrals = tf.zeros((self.num_genes, N_p))
            interactions =  tf.matmul(w, tfm.log(p_i[r]+1e-100)) + w_0[:, None]
            G = tfm.sigmoid(interactions) # TF Activation Function (sigmoid)
            sum_term = G * tfm.exp(d_j*τ)
            integrals = tf.concat([tf.zeros((self.num_genes, 1), dtype='float64'), # Trapezoid rule
                                0.5*resolution*tfm.cumsum(sum_term[:, :-1] + sum_term[:, 1:], axis=1)], axis=1) 
            exp_dt = tfm.exp(-d_j*τ)
            integrals = tfm.multiply(exp_dt, integrals)

            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            m_pred_r = b_j/d_j + tfm.multiply((a_j-b_j/d_j), exp_dt) + s_j*integrals
            m_pred = (1-mask)*m_pred + mask * m_pred_r

        return m_pred

    def get_parameters_from_state(self, all_states, state_indices,
                                  fbar=None, k_fbar=None, kbar=None, 
                                  w=None, w_0=None, σ2_m=None, Δ=None):
        k_fbar = all_states[state_indices['kinetics']][1] if k_fbar is None else k_fbar
        fbar = all_states[state_indices['latents']][0] if fbar is None else fbar
        kbar = all_states[state_indices['kinetics']][0] if kbar is None else kbar
        # w = 1*tf.ones((self.num_genes, self.num_tfs), dtype='float64') # TODO
        # w_0 = tf.zeros(self.num_genes, dtype='float64') # TODO
        w = all_states[state_indices['weights']][0] if w is None else w
        w_0 = all_states[state_indices['weights']][1] if w_0 is None else w_0
        σ2_m = all_states[state_indices['σ2_m']] if σ2_m is None else σ2_m
        Δ = None
        if self.options.delays:
            Δ = all_states[state_indices['Δ']] if Δ is None else Δ
        return fbar, k_fbar, kbar, w, w_0, σ2_m, Δ

    def genes(self, all_states=None, state_indices=None,
              k_fbar=None,
              kbar=None, 
              fbar=None, 
              w=None,
              w_0=None,
              σ2_m=None, 
              Δ=None, return_sq_diff=False):
        '''
        Computes likelihood of the genes.
        If any of the optional args are None, they are replaced by their 
        current value in all_states.
        '''
        fbar, k_fbar, kbar, w, w_0, σ2_m, Δ = self.get_parameters_from_state(
            all_states, state_indices, fbar, k_fbar, kbar, w, w_0, σ2_m, Δ)

        lik, sq_diff = self._genes(fbar, k_fbar, kbar, w, w_0, σ2_m, Δ)

        if return_sq_diff:
            return lik, sq_diff
        return lik

    @tf.function
    def _genes(self, fbar, k_fbar, kbar, wbar, w_0bar, σ2_m, Δ=None):
        m_pred = self.predict_m(kbar, k_fbar, wbar, fbar, w_0bar, Δ)
        sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred),self.data.common_indices)))
        sq_diff = tf.reduce_sum(sq_diff, axis=0)
        variance = tf.reshape(σ2_m, (-1, 1))
        if self.preprocessing_variance:
            variance = logit(variance) + self.data.σ2_m_pre # add PUMA variance
        # print(variance.shape, sq_diff.shape)
        # tf.print(variance)
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
        f_pred = inverse_positivity(fbar)
        sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))
        sq_diff = tf.reduce_sum(sq_diff, axis=0)
        log_lik = -0.5*tfm.log(2*np.pi*variance) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik, axis=1)
        if return_sq_diff:
            return log_lik, sq_diff
        return log_lik


TupleParams_pre = collections.namedtuple('TupleParams_pre', [
    'latents','σ2_m','weights', 'Δ', 'kinetics', 'σ2_f',
])
TupleParams = collections.namedtuple('TupleParams', [
    'latents','σ2_m','weights', 'Δ', 'kinetics',
])

class TranscriptionMixedSampler():
    '''
    Data is a tuple (m, f) of shapes (reps, num, time)
    time is a tuple (t, τ, common_indices)
    '''
    def __init__(self, data: DataHolder, options: Options):
        self.data = data
        self.samples = None
        self.N_p = data.τ.shape[0]
        self.N_m = data.t.shape[0]      # Number of observations

        self.num_tfs = data.f_obs.shape[1] # Number of TFs
        self.num_genes = data.m_obs.shape[1]
        self.num_replicates = data.m_obs.shape[0]

        self.likelihood = TranscriptionLikelihood(data, options)
        self.options = options
        self.kernel_selector = GPKernelSelector(data, options)

        self.state_indices = {}
        step_sizes = self.options.initial_step_sizes
        logistic_step_size = step_sizes['nuts'] if 'nuts' in step_sizes else 0.00001

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
        w_prior = LogisticNormal(f64(-1.1), f64(1.1))
        w_value = logistic(1*tf.ones((self.num_genes, self.num_tfs), dtype='float64'))
        w_0_prior = LogisticNormal(f64(-1), f64(1))
        w_0_value = 0.5*tf.ones(self.num_genes, dtype='float64')
        weights = KernelParameter('weights', [w_prior, w_0_prior], [w_value, w_0_value], step_size=logistic_step_size,
                                  hmc_log_prob=w_log_prob, requires_all_states=True)

        # Latent function & GP hyperparameters
        kernel_initial = self.kernel_selector.initial_params()

        f_step_size = step_sizes['latents'] if 'latents' in step_sizes else 20
        fbar_kernel = FKernel(data, self.likelihood, 
                              self.kernel_selector,
                              self.options.tf_mrna_present, 
                              self.state_indices,
                              f_step_size*tf.ones(self.N_p, dtype='float64'))
        fbar_initial = [
            0.3*tf.ones((self.num_replicates, self.num_tfs, self.N_p), dtype='float64'),
            *kernel_initial
        ]
        fbar = KernelParameter('latents', self.fbar_prior, fbar_initial,
                                kernel=fbar_kernel, requires_all_states=False)

        # White noise for genes
        if not options.preprocessing_variance:
            def m_sq_diff_fn(all_states):
                fbar, k_fbar, kbar, wbar, w_0bar, σ2_m, Δ = self.likelihood.get_parameters_from_state(all_states, self.state_indices)
                m_pred = self.likelihood.predict_m(kbar, k_fbar, wbar, fbar, w_0bar, Δ)
                sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred),self.data.common_indices)))
                return tf.reduce_sum(sq_diff, axis=0)

            σ2_m_kernel = GibbsKernel(data, options, self.likelihood, tfd.InverseGamma(f64(0.01), f64(0.01)), 
                                      self.state_indices, m_sq_diff_fn)
            σ2_m = KernelParameter('σ2_m', None, 1e-3*tf.ones((self.num_genes, 1), dtype='float64'), kernel=σ2_m_kernel)
        else:
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
            σ2_m = KernelParameter('σ2_m', LogisticNormal(f64(1e-5), f64(1e-4)), # f64(max(np.var(data.f_obs, axis=1)))
                            logistic(f64(5e-3))*tf.ones(self.num_genes, dtype='float64'), 
                            hmc_log_prob=σ2_m_log_prob, requires_all_states=True, step_size=logistic_step_size)

        # Kinetic parameters
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
                new_prob += tf.reduce_sum(self.params.kinetics.prior[0].log_prob(k_m))
                new_prob += tf.reduce_sum(self.params.kinetics.prior[1].log_prob(k_f))
                return tf.reduce_sum(new_prob)
            return kbar_log_prob_fn

        k_fbar_initial = 0.7*tf.ones((self.num_tfs,2), dtype='float64')
        kinetics = KernelParameter('kinetics', [LogisticNormal(0.01, 8), LogisticNormal(0.1, 5)], [kbar_initial, k_fbar_initial],
                             hmc_log_prob=kbar_log_prob, step_size=logistic_step_size, requires_all_states=True)


        delta_kernel = DeltaKernel(self.likelihood, 0, 10, self.state_indices, tfd.Exponential(f64(0.3)))
        Δ = KernelParameter('Δ', tfd.InverseGamma(f64(0.01), f64(0.01)), 0.6*tf.ones(self.num_tfs, dtype='float64'),
                        kernel=delta_kernel, requires_all_states=False)
        
        if not options.preprocessing_variance:
            def f_sq_diff_fn(all_states):
                f_pred = inverse_positivity(all_states[self.state_indices['latents']][0])
                sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))
                return tf.reduce_sum(sq_diff, axis=0)
            kernel = GibbsKernel(data, options, self.likelihood, tfd.InverseGamma(f64(0.01), f64(0.01)), 
                                 self.state_indices, f_sq_diff_fn)
            σ2_f = KernelParameter('σ2_f', None, 1e-4*tf.ones((self.num_tfs,1), dtype='float64'), kernel=kernel)
            self.params = TupleParams_pre(fbar, σ2_m, weights, Δ, kinetics, σ2_f)
        else:
            self.params = TupleParams(fbar, σ2_m, weights, Δ, kinetics)
        
        self.active_params = [
            self.params.kinetics,
            self.params.latents,
            self.params.σ2_m,
            self.params.weights
        ]
        if not options.preprocessing_variance:
            self.active_params += [self.params.σ2_f]
        if options.delays:
            self.active_params += [self.params.Δ]

        self.state_indices.update({
            param.name: i for i, param in enumerate(self.active_params)
        })

    def fbar_prior(self, fbar, param_0bar, param_1bar):
        m, K = self.kernel_selector()(param_0bar, param_1bar)
        jitter = tf.linalg.diag(1e-8 *tf.ones(self.N_p, dtype='float64'))
        prob = 0
        for r in range(self.num_replicates):
            for i in range(self.num_tfs):
                prob += tfd.MultivariateNormalTriL(loc=m, scale_tril=tf.linalg.cholesky(K[i]+jitter)).log_prob(fbar[r, i])
        return prob


    def sample(self, T=2000, store_every=10, burn_in=1000, report_every=100, num_chains=4):
        print('----- Sampling Begins -----')
        
        kernels = [param.kernel for param in self.active_params]
#         if self.options.tf_mrna_present:
        send_all_states = [param.requires_all_states for param in self.active_params]

        current_state = [param.value for param in self.active_params]
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
        for param in self.active_params:
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
        print()
        print('----- Finished -----')
        return samples, is_accepted
        
    def results(self):
        Δ = None
        σ2_f = None
        σ2_m = self.samples[self.state_indices['σ2_m']]
        if self.options.preprocessing_variance:
            σ2_m = logit(σ2_m)
        else:
            σ2_f = self.samples[self.state_indices['σ2_f']]

        kbar =   self.samples[self.state_indices['kinetics']][0].numpy()
        k_fbar = self.samples[self.state_indices['kinetics']][1].numpy()
        fbar =   self.samples[self.state_indices['latents']][0]
        kernel_params = self.samples[self.state_indices['latents']][1:]
        w =      self.samples[self.state_indices['weights']][0]
        w_0 =    self.samples[self.state_indices['weights']][1]
        if self.options.delays:
            Δ =  self.samples[self.state_indices['Δ']]
        return SampleResults(fbar, kbar, k_fbar, Δ, kernel_params, w, w_0, σ2_m, σ2_f)

    @staticmethod
    def initialise_from_state(args, state):
        model = TranscriptionMixedSampler(*args)
        model.acceptance_rates = state.acceptance_rates
        model.samples = state.samples
        return model

    def predict_m_with_results(self, results, i=1):
        delay = results.Δ[-i] if self.options.delays else None
        return self.likelihood.predict_m(results.kbar[-i], results.k_fbar[-i], results.wbar[-i], 
                                         results.fbar[-i], results.w_0bar[-i], delay)

    def predict_m_with_current(self):
        return self.likelihood.predict_m(self.params.kinetics.value[0], 
                                         self.params.kinetics.value[1], 
                                         self.params.weights.value[0], 
                                         self.params.latents.value,
                                         self.params.weights.value[1])
