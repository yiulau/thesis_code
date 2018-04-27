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
        self.beta = nn.Parameter(torch.zeros(self.dim*3+4),requires_grad=True)
        self.y = y
        self.X = X
        self.nu = nu

        # beta[:dim] = z
        # beta[(dim):(2dim)] = r1_local
        # beta[(2dim):(3dim)] = r2_local
        # beta[3dim] = r1_global
        # beta[3dim+1] = r2_global
        # beta[3dim+2] = sigma
        # beta[3dim+3] = w0
        return()

    def forward(self):
        z = self.beta[:self.dim]
        r1_local = self.beta[(self.dim):(2*self.dim)]
        r2_local = self.beta[(2*self.dim):(3*self.dim)]
        r1_global = self.beta[3*self.dim]
        r2_global = self.beta[3*self.dim+1]
        sigma = self.beta[3*self.dim+2]
        w0 = self.beta[3*self.dim+3]

        tau =  r1_global * torch.sqrt(r2_global)
        lamb = r1_local * torch.sqrt(r2_local)
        w = z * lamb * tau

        outy = (self.y - (w0 + self.X.mv(w)))*(self.y - (w0 + self.X.mv(w)))/(sigma*sigma) * 0.5
        outz = torch.dot(z,z) * 0.5
        outr1_local = torch.dot(r1_local,r1_local)
        outr2_local = ((0.5*self.nu+1)*torch.log(r2_local)  + 0.5 * self.nu/r2_local).sum()
        outr1_global = r1_global*r1_global * 0.5
        outr2_global = 1.5 * torch.log(r2_global) + 0.5/r2_global
        outw0 = w0*w0/(25.)
        out = outy+outz+outr1_local+outr2_local+outr1_global+outr2_global+outw0
        return(out)

    def load_explcit_gradient(self):
        return()