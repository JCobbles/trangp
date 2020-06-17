from reggae.models.likelihood import TranscriptionLikelihood


'''Maximum Likelihood Estimate (TODO)'''
class TranscriptionMLE():
    def __init__(self, data: DataHolder, options: Options):
        self.data = data
        self.samples = None
        self.N_p = data.τ.shape[0]
        self.N_m = data.t.shape[0]      # Number of observations

        self.num_tfs = data.f_obs.shape[1] # Number of TFs
        self.num_genes = data.m_obs.shape[1]
        self.num_replicates = data.m_obs.shape[0]

        self.likelihood = TranscriptionLikelihood(data, options)
        self.options = options
