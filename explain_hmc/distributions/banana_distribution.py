from abstract.abstract_class_V import V
import torch,math,numpy
import torch.nn as nn
from torch.autograd import Variable

precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)

class V_banana(V):
    def V_setup(self):
        self.n = 100
        y_np = numpy.random.randn(self.n)*2.+1.
        self.y = Variable(torch.from_numpy(y_np),requires_grad=False).type(precision_type)
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(2),requires_grad=True)
        #self.n = n
        # beta[n-1] = y ,
        # beta[:(n-1)] = x
        self.var_y = 2

        return()

    def forward(self):
        # returns -log posterior
        theta = self.beta
        y = self.y


        logp_y = -torch.dot(((y-theta[0])-theta[1]*theta[1]),((y-theta[0])-theta[1]*theta[1]))/(2*self.var_y*self.var_y)
        logp_theta =  -torch.dot(theta,theta)/2
        logprob = logp_y + logp_theta
        out = -logprob
        return(out)

    def load_explicit_gradient(self):
        out = torch.zeros(2)
        y = self.y.data
        theta = self.beta.data
        out[0] = -((y-theta[0])-theta[1]*theta[1]).sum()/(self.var_y*self.var_y) + theta[0]
        out[1] = -2*((y-theta[0])-theta[1]*theta[1]).sum()*theta[1]/(self.var_y*self.var_y) + theta[1]
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(2,2)
        y = self.y.data
        theta = self.beta.data
        out[0,0] = self.n/(self.var_y*self.var_y) + 1
        out[0,1] = 2*self.n*theta[1]/(self.var_y*self.var_y)
        out[1,0] = out[0,1]
        out[1,1] = -2*((y-theta[0])-3*theta[1]*theta[1]).sum()/(self.var_y*self.var_y)  + 1
        return(out)
    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives

        out = torch.zeros(2, 2, 2)
        y = self.y.data
        theta = self.beta.data

        out[0,1,1] = 2*self.n/(self.var_y*self.var_y)
        out[1,0,1] = 2*self.n/(self.var_y*self.var_y)
        out[1,1,0] = out[1,0,1]
        out[1,1,1] = 12*self.n*theta[1]/(self.var_y*self.var_y)
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