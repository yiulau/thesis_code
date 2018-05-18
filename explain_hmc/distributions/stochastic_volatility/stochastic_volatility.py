from abstract.abstract_class_V import V
import torch
import torch.nn as nn
# can implement but only do gradients
# can't do rmhmc

from torch.autograd import Variable
from explicit.general_util import logsumexp_torch

class V_stochastic_volatility(V):
    def __init__(self):
        super(V_stochastic_volatility, self).__init__()

    def V_setup(self):
        y = self.input
        self.explicit_gradient = False
        self.need_higherorderderiv = True

        self.n = len(y)
        self.dim = self.n + 3
        self.theta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        # x = self.theta[:self.n]
        # beta = (self.theta[self.n])
        # phi = sigmoid(self.theta[self.n+1])
        # inverse_logit(p) = torch.log(p/(1-p))# need stable implementation
        # sigma = exp(self.theta[self.n+2])
        self.y = Variable(torch.from_numpy(y),requires_grad=False)
        self.nu = nn.Parameter(torch.zeros())

        return()

    def forward(self):
        x = self.theta[:self.n]
        beta = self.theta[self.n]
        phi = torch.sigmoid(self.theta[self.n+1])
        sigma2 = torch.exp(self.theta[self.n+2])*torch.exp(self.theta[self.n+2])
        y_var = torch.exp(x*0.5)*beta * torch.exp(x*0.5)*beta
        x_rest_mean = phi*x[1:]
        x_rest_var = sigma2
        x_1_var = sigma2/(1.-phi*phi)
        logp_y = -(self.y * self.y / y_var).sum() * 0.5
        logp_x = - x[0]*x[0]/(x_rest_var) * 0.5 - ((x[1:]-x_rest_mean)*(x[1:]-x_rest_mean)/x_rest_var).sum() * 0.5
        log_beta = beta
        log_phi = (20-1) * torch.log(phi) + (1.5-1)*torch.log(1-phi) +torch.log(phi) + torch.log(1-phi)
        log_sigma = -(10*0.5+1)*torch.log(sigma2) -10*0.05/(2*sigma2) + 2*self.theta[self.n+2] # + log(2)
        log_posterior = logp_y + logp_x + log_beta + log_phi + log_sigma
        out = -log_posterior
        return(out)

    def load_explcit_gradient(self):
        # write down explicit gradient
        return()