import collections
from numpy import float64 as f64
from dataclasses import dataclass
from reggae.utilities import inverse_positivity, logit

GenericResults = collections.namedtuple('GenericResults', [
    'target_log_prob',
    'is_accepted',
    'acc_iter',
], defaults=[(f64(0), f64(0))])

MixedKernelResults = collections.namedtuple('MixedKernelResults', [
    'inner_results',
#     'grads_target_log_prob',
#     'step_size',
#     'log_accept_ratio',
    'is_accepted',
])

@dataclass
class SampleResults:
    fbar: object
    kbar: object
    k_fbar: object
    σ2_m: object
    kernel_params: object
    wbar: object
    w_0bar: object
    σ2_f: int = None

    @property
    def f(self):
        return inverse_positivity(self.fbar)
    @property
    def k(self):
        return logit(self.kbar).numpy()
    @property
    def k_f(self):
        return logit(self.k_fbar).numpy()
    @property
    def weights(self):
        return [logit(self.wbar), logit(self.w_0bar)]
