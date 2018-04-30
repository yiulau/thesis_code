from abstract_class_V_deprecated import V
import torch, numpy
import torch.nn as nn
from torch.autograd import Variable
from explicit.general_util import logsumexp_torch


precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)

class V_response_model(V):
    #def __init__(self):
    #    super(V_test_abstract, self).__init__()

    def V_setup(self):
        dim = 10
        y_np = numpy.random.binomial(n=1, p=0.5, size=dim)

        self.dim = dim+1
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(self.dim),requires_grad=True)
        self.y = Variable(torch.from_numpy(y_np),requires_grad=False).type(precision_type)
        # include

        return()

    def forward(self):
        y = self.y
        b = self.beta[:(self.dim-1)]
        theta = self.beta[self.dim-1]
        likelihood = (y* torch.log(torch.sigmoid(theta-b)) + (1-y)*torch.log(1-torch.sigmoid(theta-b))).sum()
        prior = -torch.dot(self.beta, self.beta)/(2*10)
        posterior = prior + likelihood
        out = -posterior
        return(out)

    def load_explicit_gradient(self):
        out = torch.zeros(self.dim)
        y = self.y.data
        b = self.beta[:(self.dim - 1)].data
        theta = numpy.asscalar(self.beta[self.dim - 1].data.numpy())
        sb= torch.sigmoid(theta - b)
        out[:(self.dim - 1)] = -y*(1-sb)+(1-y)*sb - b/10
        out[self.dim-1] = (y*(1-sb) - (1-y)*sb).sum() - theta/10
        out = -out
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.dim,self.dim)
        y = self.y.data
        b = self.beta[:(self.dim - 1)].data
        theta = self.beta[self.dim - 1].data
        sb = torch.sigmoid(theta - b)

        out[:(self.dim-1),:(self.dim-1)] = torch.diag(-sb*(1-sb)-1/10)
        out[:(self.dim-1),self.dim-1] = (sb*(1-sb))
        out[self.dim-1,self.dim-1] = (-(sb*(1-sb))).sum()-1/10
        out[self.dim-1,:(self.dim - 1)] = out[:(self.dim-1),self.dim-1]
        out = -out
        return(out)

    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.dim,self.dim,self.dim)
        y = self.y.data
        b = self.beta[:(self.dim - 1)].data
        theta = numpy.asscalar(self.beta[self.dim - 1].data.numpy())
        sb = torch.sigmoid(theta - b)
        sb_prime = sb *(1-sb)*(1-2*sb)
        # case 1

        for i in range(self.dim-1):
            out[i,i,i]= sb_prime[i]

        out[:(self.dim - 1), :(self.dim - 1), self.dim - 1] = -torch.diag(sb_prime)
        out[:(self.dim-1),self.dim-1,self.dim-1] = sb_prime
        out[:(self.dim - 1), self.dim - 1,:(self.dim - 1)] = -torch.diag(sb_prime)


        # case 2
        out[self.dim-1,:(self.dim-1),:(self.dim-1)] = -torch.diag(sb_prime)
        out[self.dim-1,:(self.dim-1),self.dim-1] = sb_prime
        out[self.dim-1,self.dim-1,self.dim-1] = -sb_prime.sum()
        out[self.dim-1,self.dim-1,:(self.dim-1)] = out[self.dim-1,:(self.dim-1),self.dim-1]

        out = -out
        return(out)