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
    def log_lik(self,x):
        return((-((x-self.mean)**2/(2*self.var))))
    def log_grad(self,x):
        return(((x-self.mean)/self.var))


class prior(likelihood):
    def log_lik(self,x):
        return(0.)
    def log_grad(self,x):
        return(0.)

post1 = posterior(hypers)
prior1 = prior()
# HMC sampling
init = 0.2
epsilon = 1.1
L = 10
chain_L = 350
hmc = hmc_sampler(post1,prior1,epsilon,L,numpy.array([init]))
#output = hmc.sample()
#print(output)
#exit()
store_array = numpy.zeros(shape=(chain_L,2))
for i in range(0,chain_L):
    jitt = numpy.random.normal(size=1)*0.01
    hmc = hmc_sampler(post1,prior1,epsilon+jitt,L,numpy.array([init]))
    output = hmc.sample()
    init = output[1]
    #print(init)
    store_array[i,0] = output[0]
    store_array[i,1] = output[1]

print(sum(store_array[:,0])/chain_L)
import matplotlib.pyplot as plt
plt.plot(range(0,chain_L),store_array[:,1],'ro')
plt.show()
#exit()
# RW Metropolis Sampling
temp_f = lambda x : post1.log_lik(x)
RW_sampler = metropolis(numpy.array([0.05]),3.21,temp_f,chain_L)
RW_sampler.sample()
print(sum(RW_sampler.store_matrix[:,0])/chain_L)
plt.plot(range(0,chain_L),RW_sampler.store_matrix[:,1],'ro')
plt.show()
