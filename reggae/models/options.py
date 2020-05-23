class Options():
    def __init__(self, 
                 preprocessing_variance=True, 
                 tf_mrna_present=True, 
                 delays=False,
                 latent_function_metropolis=True,
                 kernel='rbf',
                 initial_step_sizes={}):
        self.preprocessing_variance = preprocessing_variance
        self.tf_mrna_present = tf_mrna_present
        self.delays = delays
        self.latent_function_metropolis = latent_function_metropolis
        self.kernel = kernel
        self.initial_step_sizes = initial_step_sizes