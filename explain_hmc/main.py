from likelihood import likelihood
from hmc import hmc_sampler
import sys
import numpy
# First example draw from 1-d standard normal
hypers= {"mean":0.,"var":1.}
class posterior(likelihood):
    def __init__(self,**hypers):
        self.mean = hypers["mean"]
        self.var = hypers["var"]
    def lik(self,x):
        return(exp((x-self.mean)**2/var))
    def log_grad(self,x):
        return((2*(x-self.hyper)/var)

class prior(likelihood):
    def lik(self,x):
        return(1.)
    def log_grad(self,x):
        return(0.)
post1 = posterior(hypers)
prior1 = prior()
print(post1.lik(1))
print(post1.log_grad(1))
