from abstract.abstract_class_V import V
import abc
class new_base_V_nn_class(V):
    def __init__(self):
        super(new_base_V_nn_class, self).__init__()

    @abc.abstract_method
    def log_likelihood(self):
        pass

    def log_prior(self):
        # do something to self.list_layers
        return
    @abc.abstract_method
    def prepare_prior(self):
        pass
    def forward(self):
        out = self.log_likelihood() + self.log_prior()
        return(out)
# need to have a list of param per layer