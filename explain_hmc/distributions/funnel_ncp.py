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
        # beta[n-1] = y_raw ,
        # beta[:(n-1)] = x_raw
        return()

    def forward(self):
        x_raw = self.beta[:(self.n-1)]
        y_raw = self.beta[self.n-1]
        y = y_raw * 3.
        x = x_raw * torch.exp(y*0.5)
        out_y = y_raw * y_raw * 0.5
        out_x = (x_raw * x_raw).sum() * 0.5
        out = out_y + out_x
        return(out)

    def output_desired_data(self):
        out = torch.zeros(len(self.beta))
        x_raw = self.beta[:(self.n - 1)]
        y_raw = self.beta[self.n - 1]
        y = y_raw * 3.
        x = x_raw * torch.exp(y * 0.5)
        out[:(self.n - 1)].copy_(x.data)
        out[self.n - 1].copy_(y.data)
        return(out)
    def load_explcit_gradient(self):
        return()