from reggae.mcmc import nuts

from numpy import float64
class Parameter():
    def __init__(self, 
                 name, 
                 prior, 
                 initial_value, 
                 step_size=1., 
                 proposal_dist=None, 
                 constraint=None, 
                 fixed=False,
                 hmc_log_prob=None,
                 kernel=None):
        self.name = name
        self.prior = prior
        self.step_size = step_size
        self.proposal_dist = proposal_dist
        self.hmc_log_prob = hmc_log_prob
        if hmc_log_prob is not None:
            self.kernel = nuts.NoUTurnSampler(hmc_log_prob, step_size=step_size)
        if kernel is not None:
            self.kernel = kernel
        if constraint is None:
            self.constrained = lambda x:x
        else:
            self.constrained = constraint
        self.value = initial_value
        self.fixed = fixed

    def constrain(self, *args):
        return self.constrained(*args)

    def propose(self, *args):
        if self.fixed:
            return self.value
        assert self.proposal_dist is not None, 'proposal_dist must not be None if you use propose()'
        return self.proposal_dist(*args).sample().numpy()
