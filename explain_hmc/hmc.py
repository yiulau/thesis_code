import abc

class hmc_sampler(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,likobj):
        self.likobj= likobj
    def U(self,x):
        return(self.likobj.lik(x)+self.likobj.log_grad(x))
    @abc.abstractmethod
    def name():
        pass

          



    
