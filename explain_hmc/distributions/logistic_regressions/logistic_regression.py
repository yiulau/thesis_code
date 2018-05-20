from abstract.abstract_class_V import V
import torch, numpy
import torch.nn as nn
from torch.autograd import Variable
from explicit.general_util import logsumexp_torch


#precision_type = 'torch.DoubleTensor'
precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)

class V_logistic_regression(V):
    def __init__(self,input_npdata):
        self.y_np = input_npdata["y_np"]
        self.X_np = input_npdata["X_np"]
        super(V_logistic_regression, self).__init__()
    def V_setup(self):
        self.dim = self.X_np.shape[1]
        self.num_ob = self.X_np.shape[0]
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        self.y = Variable(torch.from_numpy(self.y_np),requires_grad=False).type(precision_type)
        self.X = Variable(torch.from_numpy(self.X_np),requires_grad=False).type(precision_type)
        # include
        self.sigma =1

        return()

    def forward(self):
        likelihood = torch.dot(self.beta, torch.mv(torch.t(self.X), self.y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(self.num_ob)), torch.mv(self.X, self.beta)))
        prior = -torch.dot(self.beta, self.beta)/(self.sigma*self.sigma) * 0.5
        posterior = prior + likelihood
        out = -posterior
        return(out)

    def load_explicit_gradient(self):
        out = torch.zeros(self.dim)
        X = self.X.data
        beta = self.beta.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        out = -X.t().mv(y - pihat) + beta
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.dim,self.dim)
        X = self.X.data
        beta = self.beta.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        out = torch.mm(X.t(), torch.mm(X.t(), torch.diag(pihat * (1. - pihat))).t()) + torch.diag(
            torch.ones(len(beta)))

        return(out)


    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.dim,self.dim,self.dim)
        X = self.X.data
        beta = self.beta.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        for i in range(self.dim):
            out[i, :, :] = (
            torch.mm(X.t(), torch.diag(pihat * (1. - pihat))).mm(torch.diag(1. - 2 * pihat) * X[:, i]).mm(X))

        return(out)

    def load_explicit_diagH(self):
        out = self.load_explicit_H()
        return (torch.diag(out))
    def load_explicit_graddiagH(self):
        temp = self.load_explicit_dH()
        out = torch.zeros(self.dim,self.dim)
        for i in range(self.dim):
            out[i,:] = torch.diag(temp[i,:,:])
        #out = out.t()
        return(out)