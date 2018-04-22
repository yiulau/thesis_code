from abstract_class_V import V
import torch
import torch.nn as nn


class V_test_abstract(V):
    def __init__(self):
        super(V_test_abstract, self).__init__()

    def V_setup(self,n):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(n),requires_grad=True)
        self.n = n
        # beta[n-1] = y ,
        # beta[:(n-1)] = x
        return()

    def forward(self):
        x = self.beta[:(self.n-1)]
        y = self.beta[self.n-1]
        out_y = y*y * 1/9.* 0.5
        out_x = (x * x)/(torch.exp(y*0.5)*torch.exp(y*0.5)) * 0.5
        out = out_y + out_x
        return(out)

    def load_explcit_gradient(self):
        return()