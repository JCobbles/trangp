class Parameter():
    def __init__(self, name, prior, initial_value, step_size=1., proposal_dist=None, constraint=None):
        self.name = name
        self.prior = prior
        self.step_size = step_size
        self.proposal_dist = proposal_dist
        if constraint is None:
            self.constrained = lambda x:x
        else:
            self.constrained = constraint
        self.value = initial_value

    def set_value(self, value):
        this.value = value
    def constrain(self, *args):
        return self.constrained(*args)
    
    def propose(self, *args):
        assert self.proposal_dist is not None, 'proposal_dist must not be None if you use propose()'
        return self.proposal_dist(*args).sample().numpy()
