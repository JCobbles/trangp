from datetime import datetime
import pickle

import tensorflow as tf
from tensorflow import math as tfm
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from reggae.mcmc import MetropolisHastings
from reggae.mcmc.parameter import KernelParameter, Parameter, Params
from reggae.models.results import GenericResults, SampleResults
from reggae.models import Options, GPKernelSelector
from reggae.mcmc.kernels import LatentKernel, MixedKernel, DelayKernel, GibbsKernel
from reggae.data_loaders import DataHolder
from reggae.utilities import rotate, jitter_cholesky, logit, logistic, LogisticNormal, inverse_positivity, save_object

import numpy as np

f64 = np.float64
PI = tf.constant(np.pi, dtype='float64')


class TranscriptionLikelihood():
    def __init__(self, data: DataHolder, options: Options):
        self.options = options
        self.data = data
        self.preprocessing_variance = options.preprocessing_variance
        self.num_genes = data.m_obs.shape[1]
        self.num_tfs = data.f_obs.shape[1]
        self.num_replicates = data.f_obs.shape[0]

    @tf.function
    def calculate_protein(self, fbar, k_fbar, Δ): # Calculate p_i vector
        τ = self.data.τ
        f_i = inverse_positivity(fbar)
        δ_i = tf.reshape(logit(k_fbar), (-1, 1))
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

        # Approximate integral (trapezoid rule)
        resolution = τ[1]-τ[0]
        sum_term = tfm.multiply(tfm.exp(δ_i*τ), f_i)
        cumsum = 0.5*resolution*tfm.cumsum(sum_term[:, :, :-1] + sum_term[:, :, 1:], axis=2)
        integrals = tf.concat([tf.zeros((self.num_replicates, self.num_tfs, 1), dtype='float64'), cumsum], axis=2) 
        exp_δt = tfm.exp(-δ_i*τ)
        p_i = exp_δt * integrals
        return p_i

    @tf.function
    def predict_m(self, kbar, k_fbar, wbar, fbar, w_0bar, Δ):
        # Take relevant parameters out of log-space
        if self.options.kinetic_exponential:
            kin = (tf.reshape(tf.exp(logit(kbar[:, i])), (-1, 1)) for i in range(kbar.shape[1]))
        else:
            kin = (tf.reshape(logit(kbar[:, i]), (-1, 1)) for i in range(kbar.shape[1]))
        if self.options.initial_conditions:
            a_j, b_j, d_j, s_j = kin
        else:
            b_j, d_j, s_j = kin
        w = logit(wbar)
        w_0 = tf.reshape(logit(w_0bar), (-1, 1))
        τ = self.data.τ
        N_p = self.data.τ.shape[0]

        p_i = inverse_positivity(fbar)
        if self.options.translation:
            p_i = self.calculate_protein(fbar, k_fbar, Δ)

        # Calculate m_pred
        resolution = τ[1]-τ[0]
        interactions =  tf.matmul(w, tfm.log(p_i+1e-100)) + w_0
        G = tfm.sigmoid(interactions) # TF Activation Function (sigmoid)
        sum_term = G * tfm.exp(d_j*τ)
        integrals = tf.concat([tf.zeros((self.num_replicates, self.num_genes, 1), dtype='float64'), # Trapezoid rule
                            0.5*resolution*tfm.cumsum(sum_term[:, :, :-1] + sum_term[:, :, 1:], axis=2)], axis=2) 
        exp_dt = tfm.exp(-d_j*τ)
        integrals = tfm.multiply(exp_dt, integrals)

        m_pred = b_j/d_j + s_j*integrals
        if self.options.initial_conditions:
            m_pred += tfm.multiply((a_j-b_j/d_j), exp_dt)
        return m_pred

    def get_parameters_from_state(self, all_states, state_indices,
                                  fbar=None, k_fbar=None, kbar=None, 
                                  wbar=None, w_0bar=None, σ2_m=None, Δ=None):
        nuts_index = 0
        kbar = all_states[state_indices['kinetics']][nuts_index] if kbar is None else kbar
        k_fbar = None
        if self.options.translation:
            nuts_index+=1
            k_fbar = all_states[state_indices['kinetics']][nuts_index] if k_fbar is None else k_fbar
        wbar = logistic(1*tf.ones((self.num_genes, self.num_tfs), dtype='float64'))
        w_0bar = 0.5*tf.ones(self.num_genes, dtype='float64')
        if self.options.weights:
            nuts_index+=1
            wbar = all_states[state_indices['kinetics']][nuts_index] if wbar is None else wbar
            w_0bar = all_states[state_indices['kinetics']][nuts_index+1] if w_0bar is None else w_0bar
        σ2_m = all_states[state_indices['σ2_m']] if σ2_m is None else σ2_m

        if fbar is None:
            fbar = all_states[state_indices['latents']]
            if self.options.joint_latent:
                fbar = fbar[0]
        Δ = tf.zeros((self.num_tfs,), dtype='float64')
        if self.options.delays:
            Δ = all_states[state_indices['Δ']] if Δ is None else Δ
        return fbar, k_fbar, kbar, wbar, w_0bar, σ2_m, Δ

    @tf.function#(experimental_compile=True)
    def genes(self, all_states=None, state_indices=None,
              k_fbar=None,
              kbar=None, 
              fbar=None, 
              wbar=None,
              w_0bar=None,
              σ2_m=None, 
              Δ=None):
        '''
        Computes likelihood of the genes.
        If any of the optional args are None, they are replaced by their 
        current value in all_states.
        '''
        fbar, k_fbar, kbar, wbar, w_0bar, σ2_m, Δ = self.get_parameters_from_state(
            all_states, state_indices, fbar, k_fbar, kbar, wbar, w_0bar, σ2_m, Δ)
        m_pred = self.predict_m(kbar, k_fbar, wbar, fbar, w_0bar, Δ)
        sq_diff = tfm.square(self.data.m_obs - tf.transpose(tf.gather(tf.transpose(m_pred),self.data.common_indices)))
        # sq_diff = tf.reduce_sum(sq_diff, axis=0)
        variance = tf.reshape(σ2_m, (-1, 1))
        if self.preprocessing_variance:
            variance = logit(variance) + self.data.σ2_m_pre # add PUMA variance
        # print(variance.shape, sq_diff.shape)
        # tf.print(variance)
        log_lik = -0.5*tfm.log(2*PI*variance) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik)
        return log_lik

    @tf.function#(experimental_compile=True)
    def tfs(self, σ2_f, fbar): 
        '''
        Computes log-likelihood of the transcription factors.
        '''
        # assert self.options.tf_mrna_present
        if not self.preprocessing_variance:
            variance = tf.reshape(σ2_f, (-1, 1))
        else:
            variance = self.data.σ2_f_pre
        f_pred = inverse_positivity(fbar)
        sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))
        log_lik = -0.5*tfm.log(2*PI*variance) - 0.5*sq_diff/variance
        log_lik = tf.reduce_sum(log_lik)

        return log_lik


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


        # Latent function & GP hyperparameters
        kernel_initial = self.kernel_selector.initial_params()

        f_step_size = step_sizes['latents'] if 'latents' in step_sizes else 20
        latents_kernel = LatentKernel(data, options, self.likelihood, 
                                      self.kernel_selector,
                                      self.state_indices,
                                      f_step_size*tf.ones(self.N_p, dtype='float64'))
        latents_initial = 0.3*tf.ones((self.num_replicates, self.num_tfs, self.N_p), dtype='float64')
        if self.options.joint_latent:
            latents_initial = [latents_initial, *kernel_initial]
        latents = KernelParameter('latents', self.fbar_prior, latents_initial,
                                kernel=latents_kernel, requires_all_states=False)

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
            σ2_m = KernelParameter('σ2_m', LogisticNormal(f64(1e-5), f64(1e-2)), # f64(max(np.var(data.f_obs, axis=1)))
                            logistic(f64(5e-3))*tf.ones(self.num_genes, dtype='float64'), 
                            hmc_log_prob=σ2_m_log_prob, requires_all_states=True, step_size=logistic_step_size)
        kernel_params = None
        if not self.options.joint_latent:
            # GP kernel
            def kernel_params_log_prob(all_states):
                def kernel_params_log_prob(param_0bar, param_1bar):
                    param_0 = logit(param_0bar, nan_replace=self.params.kernel_params.prior[0].b)
                    param_1 = logit(param_1bar, nan_replace=self.params.kernel_params.prior[1].b)
                    new_prob = tf.reduce_sum(self.params.latents.prior(
                                all_states[self.state_indices['latents']], param_0bar, param_1bar))
                    new_prob += self.params.kernel_params.prior[0].log_prob(param_0)
                    new_prob += self.params.kernel_params.prior[1].log_prob(param_1)
                    return tf.reduce_sum(new_prob)
                return kernel_params_log_prob

            kernel_initial = self.kernel_selector.initial_params()
            kernel_ranges = self.kernel_selector.ranges()
            kernel_params = KernelParameter('kernel_params', 
                        [LogisticNormal(*kernel_ranges[0]), LogisticNormal(*kernel_ranges[1])],
                        [logistic(k) for k in kernel_initial], 
                        step_size=0.1*logistic_step_size, hmc_log_prob=kernel_params_log_prob, requires_all_states=True)

        
        # Kinetic parameters & Interaction weights
        w_prior = LogisticNormal(f64(-2), f64(2))
        w_initial = logistic(1*tf.ones((self.num_genes, self.num_tfs), dtype='float64'))
        w_0_prior = LogisticNormal(f64(-1), f64(1))
        w_0_initial = 0.5*tf.ones(self.num_genes, dtype='float64')

        num_kin = 4 if self.options.initial_conditions else 3
        kbar_initial = 0.8*tf.ones((self.num_genes, num_kin), dtype='float64')

        def kbar_log_prob(all_states):
            def kbar_log_prob_fn(*args): #kbar, k_fbar, wbar, w_0bar
                index = 0
                kbar = args[index]
                new_prob = 0
                k_m =logit(kbar)
                if self.options.kinetic_exponential:
                    k_m = tf.exp(k_m)
                # tf.print(k_m)
                lik_args = {'kbar': kbar}
                new_prob += tf.reduce_sum(self.params.kinetics.prior[index].log_prob(k_m))
                # tf.print('kbar', new_prob)
                if options.translation:
                    index += 1
                    k_fbar = args[index]
                    lik_args['k_fbar'] = k_fbar
                    kfprob = tf.reduce_sum(self.params.kinetics.prior[index].log_prob(logit(k_fbar)))
                    # tf.print('k_f_', kfprob)
                    new_prob += kfprob
                if options.weights:
                    index+= 1
                    wbar, w_0bar = args[index], args[index+1]
                    lik_args['wbar'] = wbar
                    lik_args['w_0bar'] = w_0bar
                    new_prob += tf.reduce_sum(self.params.kinetics.prior[index].log_prob(logit(wbar))) 
                    new_prob += tf.reduce_sum(self.params.kinetics.prior[index+1].log_prob(logit(w_0bar)))
                
                new_prob += tf.reduce_sum(self.likelihood.genes(
                    all_states=all_states,
                    state_indices=self.state_indices,
                    **lik_args
                ))
                # tf.print('end_prob', new_prob)
                return tf.reduce_sum(new_prob)
            return kbar_log_prob_fn


        k_fbar_initial = 0.8*tf.ones((self.num_tfs,), dtype='float64')

        kinetics_initial = [kbar_initial]
        kinetics_priors = [LogisticNormal(0.01, 100)]
        if options.translation:
            kinetics_initial += [k_fbar_initial]
            kinetics_priors += [LogisticNormal(0.1, 7)]
        if options.weights:
            kinetics_initial += [w_initial, w_0_initial]
            kinetics_priors += [w_prior, w_0_prior]
        kinetics = KernelParameter(
            'kinetics', 
            kinetics_priors, 
            kinetics_initial,
            hmc_log_prob=kbar_log_prob, step_size=logistic_step_size, requires_all_states=True)


        delta_kernel = DelayKernel(self.likelihood, 0, 10, self.state_indices, tfd.Exponential(f64(0.3)))
        Δ = KernelParameter('Δ', tfd.InverseGamma(f64(0.01), f64(0.01)), 0.6*tf.ones(self.num_tfs, dtype='float64'),
                        kernel=delta_kernel, requires_all_states=False)
        
        σ2_f = None
        if not options.preprocessing_variance:
            def f_sq_diff_fn(all_states):
                f_pred = inverse_positivity(all_states[self.state_indices['latents']][0])
                sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))
                return tf.reduce_sum(sq_diff, axis=0)
            kernel = GibbsKernel(data, options, self.likelihood, tfd.InverseGamma(f64(0.01), f64(0.01)), 
                                 self.state_indices, f_sq_diff_fn)
            σ2_f = KernelParameter('σ2_f', None, 1e-4*tf.ones((self.num_tfs,1), dtype='float64'), kernel=kernel)
        
        self.params = Params(latents, None, kinetics, Δ, kernel_params, σ2_m, σ2_f)
        
        self.active_params = [
            self.params.kinetics,
            self.params.latents,
            self.params.σ2_m,
        ]
        if not options.joint_latent:
            self.active_params += [self.params.kernel_params]
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


    def sample(self, T=2000, store_every=10, burn_in=1000, report_every=100, num_chains=4, profile=False):
        print('----- Sampling Begins -----')
        
        kernels = [param.kernel for param in self.active_params]
#         if self.options.tf_mrna_present:
        send_all_states = [param.requires_all_states for param in self.active_params]

        current_state = [param.value for param in self.active_params]
        mixed_kern = MixedKernel(kernels, send_all_states, T)
        
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

        if profile:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = '.\\logs\\reggae\\%s' % stamp
            tf.profiler.experimental.start(logdir)
        samples, is_accepted = run_chain()
        if profile:
            tf.profiler.experimental.stop()

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
    
    def sample_proteins(self, results, num_results):
        p_samples = list()
        for i in range(1, num_results+1):
            p_samples.append(self.likelihood.calculate_protein(results.fbar[-i], results.k_fbar[-i], None))
        return np.array(p_samples)

    def sample_latents(self, results, num_results):
        m_preds = list()
        for i in range(1, num_results):
            m_preds.append(self.predict_m_with_results(results, i))
        return np.array(m_preds)

    def results(self):
        Δ = σ2_f = k_fbar = None
        σ2_m = self.samples[self.state_indices['σ2_m']]
        if self.options.preprocessing_variance:
            σ2_m = logit(σ2_m)
        else:
            σ2_f = self.samples[self.state_indices['σ2_f']]

        kbar = self.samples[self.state_indices['kinetics']][0].numpy()
        fbar = self.samples[self.state_indices['latents']]
        if self.options.translation:
            k_fbar = self.samples[self.state_indices['kinetics']][1].numpy()
            if k_fbar.ndim < 3:
                k_fbar = np.expand_dims(k_fbar, 2)
        if not self.options.joint_latent:
            kernel_params = self.samples[self.state_indices['kernel_params']]
        else:
            kernel_params = [fbar[1], fbar[2]]
            fbar = fbar[0]
        wbar = tf.stack([logistic(1*tf.ones((self.num_genes, self.num_tfs), dtype='float64')) for _ in range(fbar.shape[0])], axis=0)
        w_0bar = tf.stack([0.5*tf.ones(self.num_genes, dtype='float64') for _ in range(fbar.shape[0])], axis=0)
        if self.options.weights:
            wbar =      self.samples[self.state_indices['kinetics']][2]
            w_0bar =    self.samples[self.state_indices['kinetics']][3]
        if self.options.delays:
            Δ =  self.samples[self.state_indices['Δ']]
        return SampleResults(self.options, fbar, kbar, k_fbar, Δ, kernel_params, wbar, w_0bar, σ2_m, σ2_f)

    def save(self, name):
        save_object({'samples': self.samples, 'is_accepted': self.is_accepted}, f'custom-{name}')

    @staticmethod
    def load(name, args):
        model = TranscriptionMixedSampler(*args)

        import os
        path = os.path.join(os.getcwd(), 'saved_models')
        fs = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f'custom-{name}')]
        files = sorted(fs, key=os.path.getmtime)
        with open(files[-1], 'rb') as f:
            saved_model = pickle.load(f)
            model.samples = saved_model['samples']
            model.is_accepted = saved_model['is_accepted']
        for param in model.active_params:
            index = model.state_indices[param.name]
            param_samples = model.samples[index]
            if type(param_samples) is list:
                param_samples = [[param_samples[i][-1] for i in range(len(param_samples))]]

            param.value = param_samples[-1]

        return model


    @staticmethod
    def initialise_from_state(args, state):
        model = TranscriptionMixedSampler(*args)
        model.is_accepted = state.is_accepted
        model.samples = state.samples
        return model

    def predict_m_with_results(self, results, i=1):
        delay = results.Δ[-i] if self.options.delays else None
        k_fbar = results.k_fbar[-i] if self.options.translation else None
        return self.likelihood.predict_m(results.kbar[-i], k_fbar, results.wbar[-i], 
                                         results.fbar[-i], results.w_0bar[-i], delay)

    def predict_m_with_current(self):
        return self.likelihood.predict_m(self.params.kinetics.value[0], 
                                         self.params.kinetics.value[1], 
                                         self.params.weights.value[0], 
                                         self.params.latents.value,
                                         self.params.weights.value[1])
