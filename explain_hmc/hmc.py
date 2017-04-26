from math import log, exp
import numpy
def accept(curU,proU,curK,proK):
    u = numpy.random.uniform()
    return(u<exp(curU-proU+curK-proK))
class hmc_sampler(object):
    def __init__(self,likobj,prior,ep,L,init_q):
        self.likobj = likobj
        self.prior = prior
        self.ep = ep
        self.L = L
        self.q = init_q
        self.dim = init_q.shape[0] 
    def U(self,x):
        return(-self.likobj.log_lik(x)-self.prior.log_lik(x))
    
    def gradU(self,x):
        return(self.likobj.log_grad(x) + self.prior.log_grad(x))
               
    def K(self,x):
        return(sum(x**2)/2.)
    def resample(self):
        return(numpy.random.normal(size=self.dim))
    def sample(self):
        initp = self.resample()
        initq = self.q
        p = initp
        p = p - self.ep * self.gradU(self.q)/2.0
        for i in range(0,self.L):
            #print(self.q)
            #print(p)
            self.q = self.q + self.ep * p
            if i!=(self.L-1):
                p = p - self.ep * self.gradU(self.q)
        p = p - self.ep * self.gradU(self.q)/2.0
        #print(self.q)
        curU = self.U(self.q)
        curK = self.K(p)
        initU = self.U(initq)
        initK = self.K(initp)

        acceptance = accept(curU=initU,proU=curU,curK=initK,proK=curK)
        if acceptance:
            return((acceptance,self.q))
        else:
            return((acceptance,initq))


