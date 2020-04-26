import pymc3 as pm
import numpy as np
from pymc3.step_methods.metropolis import *
from pymc3.step_methods.metropolis import tune
from pymc3.theanof import floatX
from pymc3.step_methods.arraystep import metrop_select

class MetropolisCustomProposal(Metropolis):
    """
    Metropolis-Hastings sampling step
    Parameters
    ----------
    vars: list
        List of variables for sampler
    S: standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist: function
        Function that returns sample when parameterized with
        previous value and S
    scaling: scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode:  string or `Mode` instance.
        compilation mode passed to Theano functions
    """
    name = 'metropolis'

    default_blocked = False
    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'accepted': np.bool,
        'tune': np.bool,
        'scaling': np.float64,
    }]

    def __init__(self, vars=None, S=None, proposal_dist=None, scaling=1.,
                 tune=True, tune_interval=100, model=None, mode=None, **kwargs):
        super().__init__(vars=vars, S=S, proposal_dist=None, scaling=scaling,
                 tune=tune, tune_interval=tune_interval, model=model, mode=mode)
        assert proposal_dist is not None
        self.proposal_dist = proposal_dist
        self.S = S

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        q = floatX(self.proposal_dist(q0, self.S) * self.scaling)

        accept = self.delta_logp(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'scaling': self.scaling,
            'accept': np.exp(accept),
            'accepted': accepted,
        }

        return q_new, [stats]
