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
        # beta[3dim] = log(sigma)
        # beta[3dim+1] = log(tau)

        return()

    def log_likelihood(self):
        w = self.beta[:self.dim]
        lam = torch.exp(self.beta[(self.dim):(2 * self.dim)])
        eta = torch.exp(self.beta[(2 * self.dim):(3 * self.dim)])
        tau = torch.exp(self.beta[3 * self.dim])
        sigma = torch.exp(self.beta[3 * self.dim + 1])
        outy = (self.y - (self.X.mv(w))) * (self.y - (self.X.mv(w))) / (sigma * sigma) * 0.5
        return(outy)

    def log_prior(self):
        w = self.beta[:self.dim]
        lam = torch.exp(self.beta[(self.dim):(2 * self.dim)])
        eta = torch.exp(self.beta[(2 * self.dim):(3 * self.dim)])
        tau = torch.exp(self.beta[3 * self.dim])
        out_lam = ((self.nu + 1.) * 0.5 + torch.log(1 + (1 / self.nu) * (lam * lam))).sum()
        out_eta = ((self.nu + 1.) * 0.5 + torch.log(1 + (1 / self.nu) * (eta * eta))).sum()
        out_hessian = self.beta[(self.dim):(2 * self.dim)].sum() + self.beta[3 * self.dim + 1] + self.beta[
            3 * self.dim] + \
                      self.beta[(2 * self.dim):(3 * self.dim)].sum()
    def forward(self):

        out_likelihoood = self.log_likelihood()
        out_prior = self.log_prior()
        out = -out_likelihoood - out_prior
        return(out)


    def load_explcit_gradient(self):
        return()

    def load_explicit_H(self):
        # write down explicit hessian
        return()
    def load_explcit_dH(self):
        # write down explicit 3 rd derivatives
        return()