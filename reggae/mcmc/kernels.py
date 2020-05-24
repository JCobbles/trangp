from abc import abstractmethod

import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

from reggae.mcmc import MetropolisHastings, Parameter
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

            self.kernels[i].all_states_hack = current_state

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


class MetropolisKernel(tfp.mcmc.TransitionKernel):
    def __init__(self, step_size, tune_every=20):
        self.step_size = tf.Variable(step_size)
        self.tune_every = tune_every

    def metropolis_is_accepted(self, new_log_prob, old_log_prob):
        alpha = tf.math.exp(new_log_prob - old_log_prob)
        return tf.random.uniform((1,), dtype='float64') < tf.math.minimum(f64(1), alpha)
    #     if is_tensor(alpha):
    #         alpha = alpha.numpy()
    #     return not np.isnan(alpha) and random.random() < min(1, alpha)

    def one_step(self, current_state, previous_kernel_results, all_states):
        new_state, prob, is_accepted = self._one_step(current_state, previous_kernel_results, all_states)

        acc_rate, iteration = previous_kernel_results.acc_iter
        acc = acc_rate*iteration
        iteration += f64(1)
        acc_rate = tf.cond(tf.equal(is_accepted, tf.constant(True)), 
                        lambda: (acc+1)/iteration, lambda: acc/iteration)
        # tf.print(acc_rate, iteration)
        tf.cond(tf.equal(tfm.floormod(iteration, self.tune_every), 0), lambda: self.tune(acc_rate), lambda:None)

        return new_state, GenericResults(prob, is_accepted, (acc_rate, iteration)) # TODO for multiple TFs

    @abstractmethod
    def _one_step(self, current_state, previous_kernel_results, all_states):
        pass
    
    def tune(self, acc_rate):
        self.step_size.assign(tf.case([
                (acc_rate < 0.01, lambda: self.step_size * 0.1),
                (acc_rate < 0.05, lambda: self.step_size * 0.5),
                (acc_rate < 0.2, lambda: self.step_size * 0.9),
                (acc_rate > 0.95, lambda: self.step_size * 10.0),
                (acc_rate > 0.75, lambda: self.step_size * 2.0),
                (acc_rate > 0.5, lambda: self.step_size * 1.1),
            ],
            default=lambda:self.step_size
        ))
        tf.print('Updating step_size', self.step_size[0], 'due to acc rate', acc_rate, '\r', end='')
    
    def is_calibrated(self):
        return True


class FKernel(MetropolisKernel):
    def __init__(self, data,
                 likelihood, 
                 fbar_prior_params,
                 kernel_priors, 
                 tf_mrna_present, 
                 state_indices, 
                 step_size):
        self.fbar_prior_params = fbar_prior_params
        self.kernel_priors = kernel_priors
        self.num_tfs = data.f_obs.shape[1]
        self.num_genes = data.m_obs.shape[1]
        self.likelihood = likelihood
        self.tf_mrna_present = True
        self.state_indices = state_indices
        self.num_replicates = data.f_obs.shape[0]
        super().__init__(step_size, tune_every=20)

    def _one_step(self, current_state, previous_kernel_results, all_states):
        # Untransformed tf mRNA vectors F (Step 1)
        old_probs = list()
        new_state = tf.identity(current_state[0])
        new_params = []
        S = tf.linalg.diag(self.step_size)
        # MH
        m, K = self.fbar_prior_params(current_state[1], current_state[2])
        # Propose new params
        v = tfd.TruncatedNormal(current_state[1], 0.07, low=0, high=100).sample()
        l2 = tfd.TruncatedNormal(current_state[2], 0.07, low=0, high=100).sample()
        m_, K_ = self.fbar_prior_params(v, l2)

        for r in range(self.num_replicates):
            # Gibbs step
            fbar = new_state[r]
            z_i = tfd.MultivariateNormalDiag(fbar, self.step_size).sample()
            fstar = tf.zeros_like(fbar)

            for i in range(self.num_tfs):
                # Compute (K_i + S)^-1 K_i
                invKsigmaK = tf.matmul(tf.linalg.inv(K[i]+S), K[i]) 
                L = jitter_cholesky(K[i]-tf.matmul(K[i], invKsigmaK))
                # print(invKsigmaK.shape)
                c_mu = tf.matmul(z_i[i, None], invKsigmaK)
                # Compute nu = L^-1 (f-mu)
                invL = tf.linalg.inv(L)
                nu = tf.linalg.matvec(invL, fbar-c_mu)

                invKsigmaK = tf.matmul(tf.linalg.inv(K_[i]+S), K_[i]) 
                L = jitter_cholesky(K_[i]-tf.matmul(K_[i], invKsigmaK))
                c_mu = tf.matmul(z_i[i, None], invKsigmaK)
                fstar_i = tf.linalg.matvec(L, nu) + c_mu
                mask = np.zeros((self.num_tfs, 1), dtype='float64')
                mask[i] = 1
                fstar = (1-mask) * fstar + mask * fstar_i

            mask = np.zeros((self.num_replicates, 1, 1), dtype='float64')
            mask[r] = 1
            test_state = (1-mask) * new_state + mask * fstar

            new_prob = self.calculate_probability(test_state, [v, l2], all_states)
            old_prob = self.calculate_probability(new_state, [current_state[1], current_state[2]], all_states)
            #previous_kernel_results.target_log_prob #tf.reduce_sum(old_m_likelihood) + old_f_likelihood

            is_accepted = self.metropolis_is_accepted(new_prob, old_prob)
            
            prob = tf.cond(tf.equal(is_accepted, tf.constant(True)), lambda:new_prob, lambda:old_prob)


            new_state = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                                lambda:new_state, lambda:test_state)
            new_params = tf.cond(tf.equal(is_accepted, tf.constant(False)),
                                 lambda:[current_state[1], current_state[2]], lambda:[v, l2])

        return [new_state, *new_params], prob, is_accepted[0]
    
    def calculate_probability(self, fstar, kernel_params, all_states):
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
        # new_prob += tf.reduce_sum(
        #     self.kernel_priors[0].log_prob(kernel_params[0]) + \
        #     self.kernel_priors[1].log_prob(kernel_params[1])
        # )
        return new_prob

    def bootstrap_results(self, init_state, all_states):
        prob = self.calculate_probability(init_state[0], [init_state[1], init_state[2]], all_states)

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
        new_state = tf.cond(iteration_number < 200, lambda: current_state, lambda: proceed())
        return new_state, GenericResults([iteration_number+1], True)

    def bootstrap_results(self, init_state, all_states):

        return GenericResults([0], True)
    
    def is_calibrated(self):
        return True

class GibbsKernel(tfp.mcmc.TransitionKernel):
    
    def __init__(self, data, options, likelihood, prior, state_indices, sq_diff_fn):
        self.data = data
        self.options = options
        self.likelihood = likelihood
        self.prior = prior
        self.state_indices = state_indices
        self.sq_diff_fn = sq_diff_fn
        self.N_p = data.τ.shape[0]

    def one_step(self, current_state, previous_kernel_results, all_states):
        # if self.options.tf_mrna_present: # (Step 5)
        # Prior parameters
        α = self.prior.concentration
        β = self.prior.scale
        # Conditional posterior of inv gamma parameters:
        sq_diff = self.sq_diff_fn(all_states)
        α_post = α + 0.5*self.N_p
        β_post = β + 0.5*tf.reduce_sum(sq_diff)
        # print(α.shape, sq_diff.shape)
        # print('val', β_post.shape, params.σ2_m.value)
        new_state = tf.repeat(tfd.InverseGamma(α_post, β_post).sample(), sq_diff.shape[0])
        new_state = tf.reshape(new_state, (sq_diff.shape[0], 1))
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
