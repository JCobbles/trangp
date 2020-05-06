import collections

GenericResults = collections.namedtuple('GenericResults', [
    'target_log_prob',
    'is_accepted',
])

MixedKernelResults = collections.namedtuple('MixedKernelResults', [
    'inner_results',
#     'grads_target_log_prob',
#     'step_size',
#     'log_accept_ratio',
#     'is_accepted',
])
