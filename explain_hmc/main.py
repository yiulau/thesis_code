from likelihood import likelihood
from hmc import hmc_sampler
import sys
from math import exp, log
import numpy
from metropolis import metropolis 
# First example draw from 1-d standard normal
hypers= {"mean":0.,"var":1.}
class posterior(likelihood):
    def __init__(self,hypers):
        self.mean = hypers["mean"]
        self.var = hypers["var"]
    def lik(self,x):
        return(exp((x-self.mean)**2/self.var))
    def log_grad(self,x):
        return((2*(x-self.mean)/self.var))


class prior(likelihood):
    def lik(self,x):
        return(1.)
    def log_grad(self,x):
        return(0.)

post1 = posterior(hypers)
prior1 = prior()

# HMC sampling
init = 0.25
epsilon = 0.03
L = 30
chain_L = 100
store_array = numpy.zeros(shape=(chain_L,2))
for i in range(0,chain_L):
    hmc = hmc_sampler(post1,prior1,epsilon,L,numpy.array([init]))
    output = hmc.sample()
    init = output[1]
    store_array[i,0] = output[0]
    store_array[i,1] = output[1]

import matplotlib.pyplot as plt
plt.plot(range(0,chain_L),store_array[:,1],'ro')
plt.show()

# RW Metropolis Sampling
temp_f = lambda x : post1.lik(x)
RW_sampler = metropolis(numpy.array([0.25]),0.21,temp_f,100)
RW_sampler.sample()
plt.plot(range(0,chain_L),RW_sampler.store_matrix[:,1],'ro')
plt.show()
