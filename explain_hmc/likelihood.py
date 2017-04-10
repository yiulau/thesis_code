import abc

class likelihood(object):
    """likelihood that returns all info about target"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def lik(self):
        pass
    @abc.abstractmethod
    def log_grad(self):
        pass


