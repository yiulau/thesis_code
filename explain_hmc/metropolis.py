import numpy 
from math import exp, log, sqrt
class metropolis(object):
    def __init__(self,initq,sigma2,target,chain_L):
        self.sigma2 = sigma2
        self.target = target
        self.q = initq
        self.store_matrix = numpy.zeros(shape=(chain_L,2))
        self.chain_L=chain_L 

    def sample(self):
        for i in range(0,self.chain_L):
            curq = self.q
            propq = sqrt(self.sigma2)*numpy.random.normal(size=curq.shape) + curq
            acceptance_prob = min(1,exp(self.target(propq)-self.target(curq)))
            accept = numpy.random.uniform()<acceptance_prob
            if accept:
                newq = propq
            else:
                newq = curq
            self.q = newq
            self.store_matrix[i,0] = accept
            self.store_matrix[i,1] = newq
        return
