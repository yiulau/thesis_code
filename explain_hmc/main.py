import abc
from likelihood import likelihood
from hmc import hmc_sampler
class test(likelihood):
    
    def lik(self,input):
        return(input+1)
    def log_grad(self,input):
        return(input+2)

x = test()
#print(x.lik(2))

class sampler_1(hmc_sampler):
    def name(self):
        pass

y = sampler_1(x)

print(y.U(1))



