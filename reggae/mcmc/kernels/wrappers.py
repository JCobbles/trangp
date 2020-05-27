import tensorflow_probability as tfp

class ESSWrapper(tfp.experimental.mcmc.EllipticalSliceSampler):
    def __init__(self, normal_sampler_fn, log_likelihood_fn):
        self.normal_sampler_fn_fn = normal_sampler_fn
        self.log_likelihood_fn_fn = log_likelihood_fn
        # Temporarily set the following
        self.normal_sampler_fn_called = normal_sampler_fn
        self.log_likelihood_fn_called = log_likelihood_fn
        super().__init__(normal_sampler_fn, log_likelihood_fn)

    @property
    def normal_sampler_fn(self):
        return self.normal_sampler_fn_called
    @property
    def log_likelihood_fn(self):
        return self.log_likelihood_fn_called

    def one_step(self, current_state, previous_kernel_results, all_states):
        self.normal_sampler_fn_called = self.normal_sampler_fn_fn(all_states)
        self.log_likelihood_fn_called = self.log_likelihood_fn_fn(all_states)
        return super().one_step(current_state, previous_kernel_results)

    def bootstrap_results(self, init_state, all_states):
        self.normal_sampler_fn_called = self.normal_sampler_fn_fn(all_states)
        self.log_likelihood_fn_called = self.log_likelihood_fn_fn(all_states)

        return super().bootstrap_results(init_state)

class NUTSWrapperKernel(tfp.mcmc.NoUTurnSampler):
    def __init__(self, target_log_prob_fn, step_size):
        self.target_log_prob_fn_fn = target_log_prob_fn
        # Temporarily set the following
        self.target_log_prob_fn_called = target_log_prob_fn
        super().__init__(target_log_prob_fn, step_size)

    @property
    def target_log_prob_fn(self):
        return self.target_log_prob_fn_called

    def one_step(self, current_state, previous_kernel_results, all_states):
        self.target_log_prob_fn_called = self.target_log_prob_fn_fn(all_states)
        return super().one_step(current_state, previous_kernel_results)

    def bootstrap_results(self, init_state, all_states):
        self.target_log_prob_fn_called = self.target_log_prob_fn_fn(all_states)
        return super().bootstrap_results(init_state)
