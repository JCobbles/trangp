from dataclasses import dataclass, field
@dataclass
class Options:
    preprocessing_variance: bool = True
    tf_mrna_present:        bool = True
    delays:                 bool = False
    kernel:                 str = 'rbf'
    latent_sampler:         str = 'joint'
    initial_step_sizes:     dict = field(default_factory=dict)
