from abstract_class_V import V
import torch
import torch.nn as nn


class V_test_abstract(V):
    def __init__(self):
        super(V_test_abstract, self).__init__()

    def V_setup(self,y,X,nu):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.dim = X.shape[1]
        self.beta = nn.Parameter(torch.zeros(self.dim*3+2),requires_grad=True)
        self.y = y
        self.X = X
        self.nu = nu

        # beta[:dim] = w
        # beta[(dim):(2dim)] = log(lam)
        # beta[(2dim):(3dim)] = log(eta)
        # beta[3dim] = sigma
        # beta[3dim+1] = log(tau)

        return()

    def forward(self):
        w = self.beta[:self.dim]
        lam = torch.exp(self.beta[(self.dim):(2*self.dim)])
        eta = torch.exp(self.beta[(2*self.dim):(3*self.dim)])
        sigma = self.beta[3*self.dim]
        tau = torch.exp(self.beta[3*self.dim+1])
        outy = (self.y - (self.X.mv(w)))*(self.y - (self.X.mv(w)))/(sigma*sigma) * 0.5
        outw = (w * w /(eta*eta*tau*tau*lam*lam)).sum() * 0.5
        out_lam = ((self.nu +1. )*0.5 + torch.log(1+ (1/self.nu)* (lam*lam))).sum()
        out_eta = ((self.nu +1. )*0.5 + torch.log(1+ (1/self.nu)* (eta*eta))).sum()
        out_hessian = self.beta[(self.dim):(2 * self.dim)].sum() + self.beta[3 * self.dim + 1] + \
                      self.beta[(2*self.dim):(3*self.dim)].sum()
        out = outy + outw + out_lam + out_eta + out_hessian
        return(out)

    def load_explcit_gradient(self):
        return()