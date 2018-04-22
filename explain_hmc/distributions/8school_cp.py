from abstract_class_V import V
import torch
import torch.nn as nn
from torch.autograd import Variable


class V_test_abstract(V):
    def __init__(self):
        super(V_test_abstract, self).__init__()

    def V_setup(self,y,sigma):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.J = len(y)
        self.beta = nn.Parameter(torch.zeros(self.J),requires_grad=True)
        # beta[J-2] = log(tau) , beta[J-1]= mu
        self.y = Variable(torch.from_numpy(y),requires_grad=False)
        self.sigma = Variable(torch.from_numpy(sigma),requires_grad=False)
        return()

    def forward(self):
        #temp =self.y - self.beta[:(self.J-2)]
        tau = torch.exp(self.beta[self.J-2])
        mu = self.beta[self.J-1]
        theta = self.beta[:(self.J-2)]
        out_y = ((self.y - theta) * (self.y-theta) / self.sigma*self.sigma).sum() * 0.5

        out_theta = ((theta-mu) * (theta-mu) / tau*tau) * 0.5 + self.beta[self.J-2]
        out_mu = mu*mu * 0.5 * 1/25.
        out_tau =  torch.log(1+ (tau/5)*(tau/5))
        out = out_y + out_theta + out_mu + out_tau
        return(out)

    def load_explcit_gradient(self):
        return()

