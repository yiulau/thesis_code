from likelihood import likelihood
from hmc import hmc_sampler
import sys
import numpy
# First example draw from 1-d standard normal
hypers= {"mean":0.,"var":1.}
print(hypers["mean"])
exit()
class posterior(likelihood):
    def __init__(self,**kwargs):
        self.mean = hypers["mean"]
        self.var = hypers["var"]
    def lik(self,x):
        return(exp((x-self.mean)**2/var))
    def log_grad(self,x):
        return((2*(x-self.hyper)/var)
print hypers
#post1 = posterior(**hypers)
