from abstract_class_V_deprecated import V
import torch,numpy
import torch.nn as nn
from torch.autograd import Variable
from explicit.general_util import logsumexp_torch

precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)

class V_hierarchical_logistic(V):
    #def __init__(self):
    #    super(V_test_abstract, self).__init__()
    # def V_setup(self,y,X,lamb)
    def V_setup(self):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        dim = 10
        num_ob = 20
        y_np = numpy.random.binomial(n=1, p=0.5, size=num_ob)
        X_np = numpy.random.randn(num_ob, dim)
        self.dim = X_np.shape[1]+1
        num_ob = X_np.shape[0]
        self.num_ob = X_np.shape[0]

        self.beta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        # sigma mapped to log space beecause we want it unconstrained
        # self.beta[self.dim] = log(sigma)
        #self.logsigma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.y = Variable(torch.from_numpy(y_np),requires_grad=False).type(precision_type)
        self.X = Variable(torch.from_numpy(X_np),requires_grad=False).type(precision_type)
        # parameter for hyperprior distribution
        self.lamb = 1
        return()

    def forward(self):
        beta = self.beta[:(self.dim-1)]
        sigma = torch.exp(self.beta[self.dim-1])
        likelihood = torch.dot(beta, torch.mv(torch.t(self.X), self.y)) - \
                     torch.sum(logsumexp_torch(Variable(torch.zeros(self.num_ob)), torch.mv(self.X, beta)))
        prior = -torch.dot(beta, beta)/(sigma*sigma) * 0.5 - self.num_ob* 0.5 * torch.log(sigma*sigma) - sigma*self.lamb

        hessian_term = -self.beta[self.dim-1]
        posterior = prior + likelihood + hessian_term
        out = -posterior
        return(out)

    def load_explicit_gradient(self):
        # write down explicit gradient
        out = torch.zeros(self.dim)
        beta = self.beta[:(self.dim - 1)].data
        sigma = numpy.asscalar(torch.exp(self.beta[self.dim - 1]).data.numpy())
        logsigma = numpy.asscalar((self.beta[self.dim - 1]).data.numpy())
        X = self.X.data
        y = self.y.data
        pihat = torch.sigmoid(torch.mv(X, beta))
        out[:(self.dim-1)] = -X.t().mv(y - pihat) + beta/(sigma*sigma)
        out[self.dim-1] = -torch.dot(beta,beta)/(sigma*sigma) + self.num_ob + sigma + 1
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.dim, self.dim)
        X = self.X.data
        y = self.y.data
        beta = self.beta[:(self.dim - 1)].data
        sigma = numpy.asscalar(torch.exp(self.beta[self.dim - 1]).data.numpy())
        pihat = torch.sigmoid(torch.mv(X, beta))
        out[:(self.dim-1),:(self.dim-1)] = torch.mm(X.t(), torch.mm(X.t(), torch.diag(pihat * (1. - pihat))).t()) + torch.diag(
            torch.ones(len(beta))*1/(sigma*sigma))
        out[:(self.dim-1),self.dim-1] = -2*beta/(sigma*sigma)
        out[self.dim-1,self.dim-1] = torch.dot(beta,beta)*2/(sigma*sigma) + sigma
        out[self.dim-1,:(self.dim - 1)].copy_(out[:(self.dim-1),self.dim-1])
        return(out)
    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.dim, self.dim, self.dim)
        X = self.X.data
        y = self.y.data
        beta = self.beta[:(self.dim - 1)].data
        sigma = numpy.asscalar(torch.exp(self.beta[self.dim - 1]).data.numpy())
        pihat = torch.sigmoid(torch.mv(X, beta))
        # case 1
        # dbeta dxidxj
        for i in range(self.dim-1):
            out[i, :(self.dim-1),:(self.dim-1)] = (
            torch.mm(X.t(), torch.diag(pihat * (1. - pihat))).mm(torch.diag(1. - 2 * pihat) * X[:, i]).mm(X))
            out[i,i,self.dim-1] = -2/(sigma*sigma)
            out[i,  self.dim - 1,i] = -2/(sigma*sigma)
            out[i,self.dim-1,self.dim-1] = 4*beta[i]/(sigma*sigma)

        # case 2
        # dsigma' dxidxj
        out[self.dim-1,: (self.dim - 1),:(self.dim - 1)].copy_(torch.eye(self.dim-1,self.dim-1)*(-2/(sigma*sigma)))
        out[self.dim-1,:(self.dim - 1),self.dim-1] = 4*beta/(sigma*sigma)
        out[self.dim - 1,  self.dim - 1,:(self.dim - 1)] = 4*beta/(sigma*sigma)
        out[self.dim-1,self.dim-1,self.dim-1] = -4* torch.dot(beta,beta)/(sigma*sigma) + sigma

        return(out)