import abc
class prior_class(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass

    def create_hyper_par_fun(self,obj):
        for block in obj.divisble_blocks:
            for name, param in block:
                setattr(obj,name)

    @abc.abstractmethod
    def prior_forward(self):
        pass
def generate_prior_dict():
    def create_hyper_par_fun(obj):
