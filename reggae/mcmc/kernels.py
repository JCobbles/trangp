import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

from reggae.mcmc import MetropolisHastings, Parameter, MetropolisKernel
from reggae.models.results import GenericResults, MixedKernelResults
from reggae.utilities import jitter_cholesky, logit

import numpy as np

f64 = np.float64

from inspect import signature

class MixedKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, kernels, send_all_states):
        '''
        send_all_states is a boolean array of size |kernels| indicating which components of the state
        have kernels whose log probability depends on the state of others, in which case MixedKernel
        will recompute the previous target_log_prob before handing it over in the `one_step` call.
        '''
        self.kernels = kernels
        self.send_all_states = send_all_states
        self.num_kernels = len(kernels)
        self.one_step_receives_state = [len(signature(k.one_step).parameters)>2 for k in kernels]
        super().__init__()

    def one_step(self, current_state, previous_kernel_results):
#         print('running', current_state, previous_kernel_results)
        new_state = list()
        is_accepted = list()
        inner_results = list()
        for i in range(self.num_kernels):
            if self.send_all_states[i]:
                wrapped_state_i = current_state[i]
                if type(wrapped_state_i) is not list:
                    wrapped_state_i = [wrapped_state_i]

                previous_kernel_results.inner_results[i] = previous_kernel_results.inner_results[i]._replace(
                    target_log_prob=self.kernels[i].target_log_prob_fn(current_state)(*wrapped_state_i))

            # if type(current_state[i]) is list:
            #     self.kernels[i].all_states_hack = [tf.identity(res) for res in current_state]
            # else:
            self.kernels[i].all_states_hack = current_state
            # if i == 3:
            #     tf.print('------')
            #     tf.print(previous_kernel_results.inner_results[i].target_log_prob)
            #     tf.print('current logit()', logit(current_state[3]))

            args = []
            try:
                if self.one_step_receives_state[i]:
                    args = [current_state]

                # state_chained = tf.expand_dims(current_state[i], 0)
                # print(state_chained)
                result_state, kernel_results = self.kernels[i].one_step(
                    current_state[i], previous_kernel_results.inner_results[i], *args)
            except Exception as e:
                tf.print('Failed at ', i, self.kernels[i], current_state)
                raise e
#                 print(result_state, kernel_results)

            # if i == 3:
            #     tf.print(kernel_results.target_log_prob, kernel_results)
            #     tf.print('=======')
            if type(result_state) is list:
                new_state.append([tf.identity(res) for res in result_state])
            else:
                new_state.append(result_state)

            is_accepted.append(kernel_results.is_accepted)
            inner_results.append(kernel_results)
        
        return new_state, MixedKernelResults(inner_results, is_accepted)

    def bootstrap_results(self, init_state):
        """Returns an object with the same type as returned by `one_step(...)[1]`.
        Args:
        init_state: `Tensor` or Python `list` of `Tensor`s representing the
        initial state(s) of the Markov chain(s).
        Returns:
        kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
        `Tensor`s representing internal calculations made within this function.
        """
        inner_kernels_bootstraps = list()
        is_accepted = list()
        for i in range(self.num_kernels):
            self.kernels[i].all_states_hack = init_state

            if hasattr(self.kernels[i], 'inner_kernel'):
                self.kernels[i].inner_kernel.all_states_hack = init_state
            if self.one_step_receives_state[i]:
                results = self.kernels[i].bootstrap_results(init_state[i], init_state)
                inner_kernels_bootstraps.append(results)
                    
            else:
                results = self.kernels[i].bootstrap_results(init_state[i])
                inner_kernels_bootstraps.append(results)
            is_accepted.append(results.is_accepted)

        return MixedKernelResults(inner_kernels_bootstraps, is_accepted)

    def is_calibrated(self):
        return True

class FKernel(MetropolisKernel):
    def __init__(self, 
                 likelihood, 
                 fbar_prior_params, 
                 num_tfs, num_genes, 
                 tf_mrna_present, 
                 state_indices, 
                 step_size):
        self.fbar_prior_params = fbar_prior_params
        self.num_tfs = num_tfs
        self.num_genes = num_genes
        self.likelihood = likelihood
        self.tf_mrna_present = True
        self.state_indices = state_indices
        self.h_f = step_size
        
    def one_step(self, current_state, previous_kernel_results, all_states):
        # Untransformed tf mRNA vectors F (Step 1)
        fbar = current_state
        old_probs = list()
        # TODO check works with multiple TFs
        # Gibbs step
        z_i = tfd.MultivariateNormalDiag(fbar, self.h_f).sample()

        # MH
        kernel_params = (all_states[self.state_indices['kernel_params']][0], all_states[self.state_indices['kernel_params']][1])
        m, K = self.fbar_prior_params(*kernel_params)
        for i in range(self.num_tfs):
            invKsigmaK = tf.matmul(tf.linalg.inv(K[i]+tf.linalg.diag(self.h_f)), K[i]) # (C_i + hI)C_i
            L = jitter_cholesky(K[i]-tf.matmul(K[i], invKsigmaK))
            c_mu = tf.matmul(z_i[i, None], invKsigmaK)
            fstar_i = tf.matmul(tf.random.normal((1, L.shape[0]), dtype='float64'), L) + c_mu
            mask = np.zeros((self.num_tfs, 1), dtype='float64')
            mask[i] = 1
            fstar = (1-mask) * fbar + mask * fstar_i
            new_prob = self.calculate_probability(fstar, all_states)
            old_prob = self.calculate_probability(fbar, all_states)
            #previous_kernel_results.target_log_prob #tf.reduce_sum(old_m_likelihood) + old_f_likelihood

            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            
            prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)


            fbar = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                            lambda:fbar, lambda:fstar)

        return fbar, GenericResults(prob, is_accepted[0]) # TODO for multiple TFs
    
    def calculate_probability(self, fstar, all_states):
        new_m_likelihood = self.likelihood.genes(
            all_states,
            self.state_indices,
            fbar=fstar,
        )
        new_f_likelihood = tf.cond(tf.equal(self.tf_mrna_present, tf.constant(True)), 
                                   lambda:tf.reduce_sum(self.likelihood.tfs(
                                       1e-6*tf.ones(self.num_tfs, dtype='float64'), # TODO
                                       fstar
                                   )), lambda:f64(0))
        new_prob = tf.reduce_sum(new_m_likelihood) + new_f_likelihood
        return new_prob

    def bootstrap_results(self, init_state, all_states):
        prob = self.calculate_probability(init_state, all_states)

        return GenericResults(prob, True)
    
    def is_calibrated(self):
        return True

class DeltaKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, likelihood, lower, upper, state_indices, prior):
        self.likelihood = likelihood
        self.state_indices = state_indices
        self.lower = lower
        self.upper = upper
        self.prior = prior
        
    def one_step(self, current_state, previous_kernel_results, all_states):
        iteration_number = previous_kernel_results.target_log_prob[0] #just roll with it

        def proceed():
            num_tfs = current_state.shape[0]
            new_state = current_state
            Δrange = np.arange(self.lower, self.upper+1, dtype='float64')
            Δrange_tf = tf.range(self.lower, self.upper+1, dtype='float64')
            for i in range(num_tfs):
                # Generate normalised cumulative distribution
                probs = list()
                mask = np.zeros((num_tfs, ), dtype='float64')
                mask[i] = 1
                
                for Δ in Δrange:
                    test_state = (1-mask) * new_state + mask * Δ

                    # if j == 0:
                    #     cumsum.append(tf.reduce_sum(self.likelihood.genes(
                    #         all_states=all_states, 
                    #         state_indices=self.state_indices,
                    #         Δ=test_state,
                    #     )) + tf.reduce_sum(self.prior.log_prob(Δ)))
                    # else:
                    probs.append(tf.reduce_sum(self.likelihood.genes(
                        all_states=all_states, 
                        state_indices=self.state_indices,
                        Δ=test_state,
                    ))) #+ tf.reduce_sum(self.prior.log_prob(Δ)))
                # curri = tf.cast(current_state[i], 'int64')
                # start_index = tf.reduce_max([self.lower, curri-2])
                # probs = tf.gather(probs, tf.range(start_index, 
                #                                   tf.reduce_min([self.upper+1, curri+3])))
                probs =  tf.stack(probs) - tfm.reduce_max(probs)
                probs = tfm.exp(probs)
                probs = probs / tfm.reduce_sum(probs)
                cumsum = tfm.cumsum(probs)
                # tf.print(cumsum)
                u = np.random.uniform()
                index = tf.where(cumsum == tf.reduce_min(cumsum[(cumsum - u) > 0]))
                chosen = Δrange_tf[index[0][0]]
                new_state = (1-mask) * new_state + mask * chosen
            return new_state
#         tf.print('final chosen state', new_state)
        new_state = tf.cond(iteration_number < 500, lambda: current_state, lambda: proceed())
        return new_state, GenericResults([iteration_number+1], True)

    def bootstrap_results(self, init_state, all_states):

        return GenericResults([0], True)
    
    def is_calibrated(self):
        return True

class GibbsKernel(tfp.mcmc.TransitionKernel):
    
    def __init__(self, data, options, likelihood, prior, state_indices):
        self.data = data
        self.options = options
        self.likelihood = likelihood
        self.prior = prior
        self.state_indices = state_indices

    def one_step(self, current_state, previous_kernel_results, all_states):
        # if self.options.tf_mrna_present: # (Step 5)
        # Prior parameters
        α = self.prior.concentration
        β = self.prior.scale
        # Conditional posterior of inv gamma parameters:
        f_pred = tfm.log(1+tfm.exp(all_states[self.state_indices['fbar']]))
        sq_diff = tfm.square(self.data.f_obs - tf.transpose(tf.gather(tf.transpose(f_pred),self.data.common_indices)))

        α_post = α + 0.5*f_pred.shape[1]
        β_post = β + 0.5*tf.reduce_sum(sq_diff)
        # print(α.shape, sq_diff.shape)
        # print('val', β_post.shape, params.σ2_m.value)
        new_state = tf.repeat(tfd.InverseGamma(α_post, β_post).sample(), f_pred.shape[0])
        new_state = tf.reshape(new_state, (f_pred.shape[0], 1))
        return new_state, GenericResults(list(), True)

    def bootstrap_results(self, init_state, all_states):
        return GenericResults(list(), True) 
    
    def is_calibrated(self):
        return True

class KbarKernel(MetropolisKernel):
    def __init__(self, likelihood, prop_dist, prior_dist, num_genes, state_indices):
        self.prop_dist = prop_dist
        self.prior_dist = prior_dist
        self.num_genes = num_genes
        self.likelihood = likelihood
        self.state_indices = state_indices
        
    def one_step(self, current_state, previous_kernel_results, all_states):

        kbar = current_state
        kstar = tf.identity(kbar)
        old_probs = list()
        is_accepteds = list()
        for j in range(self.num_genes):
            sample = self.prop_dist(kstar[j]).sample()
#             sample = params.kbar.constrain(sample, j)
            kstar = tf.concat([kstar[:j], [sample], kstar[j+1:]], axis=0)
            
            new_prob = self.likelihood.genes(
                all_states,
                self.state_indices,
                kbar=kstar, 
            )[j] + tf.reduce_sum(self.prior_dist.log_prob(sample))
            
            old_prob = previous_kernel_results.target_log_prob[j] #old_m_likelihood[j] + sum(params.kbar.prior.log_prob(kbar[j]))

            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            is_accepteds.append(is_accepted)
            
            prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)
            kstar = tf.cond(tf.equal(is_accepted, tf.constant(False)), 
                                     lambda:tf.concat([kstar[:j], [current_state[j]], kstar[j+1:]], axis=0), lambda:kstar)
            old_probs.append(prob)

        return kstar, GenericResults(old_probs, True) #TODO not just return true

    def bootstrap_results(self, init_state, all_states):
        probs = list()
        for j in range(self.num_genes):
            prob = self.likelihood.genes(
                all_states,
                self.state_indices,
                kbar=init_state, 
            )[j] + tf.reduce_sum(self.prior_dist.log_prob(init_state[j]))
            probs.append(prob)

        return GenericResults(probs, True) #TODO automatically adjust
    
    def is_calibrated(self):
        return True