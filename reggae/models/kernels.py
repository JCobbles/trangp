import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

from reggae.mcmc import MetropolisHastings, Parameter, MetropolisKernel
from reggae.models.results import GenericResults, MixedKernelResults
from reggae.utilities import get_rbf_dist, exp, mult, jitter_cholesky

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

            self.kernels[i].all_states_hack = current_state

            # if hasattr(self.kernels[i], 'inner_kernel'):
            #     self.kernels[i].inner_kernel.all_states_hack = current_state
            
            try:
                if self.one_step_receives_state[i]:
                    result_state, kernel_results = self.kernels[i].one_step(
                        current_state[i], previous_kernel_results.inner_results[i], current_state)
                else:
                    result_state, kernel_results = self.kernels[i].one_step(
                        current_state[i], previous_kernel_results.inner_results[i])
            except Exception as e:
                tf.print('Failed at ', i, self.kernels[i], current_state)
                raise e
#                 print(result_state, kernel_results)

            if type(result_state) is list: # Fix since list states don't copy, they reference
                new_state.append([tf.identity(res) for res in result_state])
            else:
                new_state.append(result_state)
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
        z_i = tf.reshape(tfd.MultivariateNormalDiag(fbar, self.h_f).sample(), (1, -1))
        # MH
        rbf_params = (all_states[self.state_indices['rbf_params']][0], all_states[self.state_indices['rbf_params']][1])
        m, K = self.fbar_prior_params(*rbf_params)
        invKsigmaK = tf.matmul(tf.linalg.inv(K+tf.linalg.diag(self.h_f)), K) # (C_i + hI)C_i
        L = jitter_cholesky(K-tf.matmul(K, invKsigmaK))
        c_mu = tf.matmul(z_i, invKsigmaK)
        fstar = tf.matmul(tf.random.normal((1, L.shape[0]), dtype='float64'), L) + c_mu

        new_prob = self.calculate_probability(fstar, all_states)
        old_prob = previous_kernel_results.target_log_prob #tf.reduce_sum(old_m_likelihood) + old_f_likelihood

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
                                       1e-4*tf.ones(self.num_tfs, dtype='float64'), # TODO
                                       fstar
                                   )), lambda:f64(0))
        new_prob = tf.reduce_sum(new_m_likelihood) + new_f_likelihood
        return new_prob

    def bootstrap_results(self, init_state, all_states):
        prob = self.calculate_probability(init_state, all_states)

        return GenericResults(prob, True) #TODO automatically adjust
    
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
