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
        return(-(log(self.likobj.lik(x))+log(self.prior.lik(x))))
    
    def gradU(self,x):
        return(self.likobj.log_grad(x) + self.prior.log_grad(x))
               
    def K(self,x):
        return(sum(x**2)/2.)
    def resample(self):
        return(numpy.random.normal(size=self.dim))
    def sample(self):
        self.initp = self.resample()
        self.p = self.initp
        self.p = self.p - self.ep * self.gradU(q)/2.0
        for i in range(0,L):
            self.q = self.q + self.ep * p
            if i!=L:
                self.p = self.p - self.ep * self.gradU(q)
        self.p = self.p - self.ep * self.gradU(q)/2.0
        self.curU = self.U(q)
        self.curK = self.K(q)
        self.initU = self.U(self.initq)
        self.initK = self.K(self.initp)

        acceptance = accept_f(self.curU,self.curK,self.initU,self.initK)
        if acceptance:
            return((acceptance,self.q))
        else:
            return((acceptance,self.initq))


