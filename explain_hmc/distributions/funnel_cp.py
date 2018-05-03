from abstract.abstract_class_V import V
import torch,math
import torch.nn as nn

precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)


class V_funnel(V):
    def V_setup(self):
        self.n = 10
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.zeros(10),requires_grad=True)
        #self.n = n
        # beta[n-1] = y ,
        # beta[:(n-1)] = x
        return()

    def forward(self):
        # returns -log posterior
        x = self.beta[:(self.n-1)]
        y = self.beta[self.n-1]
        logp_y = -y*y * 1/9.* 0.5
        #print(self.beta.data)
        #print("bottom is {}".format(torch.exp(y*0.5)*torch.exp(y*0.5)))
        logp_x = -(torch.dot(x,x))/(torch.exp(y)) * 0.5 -0.5*(self.n-1)*y
        #print("x is {}".format(x))
        #print("p_x is {}".format(p_x))
        logprob = logp_y + logp_x
        out = -logprob

        #print("logpx {}".format(logp_x.data))
        #print("-logprob {}".format(out.data))
        return(out)

    def load_explicit_gradient(self):
        out = torch.zeros(self.n)
        x = self.beta[:(self.n - 1)].data
        y = self.beta[self.n - 1].data
        #print("y is {}".format(y))
        out[:(self.n - 1)] = x/(math.exp(y))
        out[self.n-1:self.n] = (y/9 - torch.dot(x,x)/(2*torch.exp(y)) + 0.5*(self.n-1))
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.n,self.n)
        x = self.beta[:(self.n - 1)].data
        y = self.beta[self.n - 1].data
        #print("y is {}".format(y))
        #print(1/(math.exp(y/2)*math.exp(y/2)))
        out[:(self.n - 1),:(self.n - 1)]  = 1/(math.exp(y)) * torch.eye(self.n-1,self.n-1)
        out[:(self.n-1),self.n-1] = -x/(math.exp(y))
        out[self.n-1,self.n-1] = 1/9 + 1/2 * torch.dot(x,x)/(math.exp(y))
        out[self.n - 1,:(self.n - 1)].copy_(out[:(self.n-1),self.n-1])
        return(out)
    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.n,self.n,self.n)
        x = self.beta[:(self.n - 1)].data
        y = self.beta[self.n - 1].data

        #case 1

        for i in range(self.n-1):
            out[i,self.n-1,i] = -1/( math.exp(y))
            out[i, i,self.n-1] = -1 / ( math.exp(y) )
            out[i,self.n-1,self.n-1] = x[i]/ ( math.exp(y ) )
        #case 2
        case2matrix = torch.zeros(self.n,self.n)
        case2matrix[:(self.n - 1), :(self.n - 1)] =  -1/(math.exp(y)) * torch.eye(self.n-1,self.n-1)
        case2matrix[:(self.n-1),self.n-1] = x/(math.exp(y))
        case2matrix[self.n-1,self.n-1] =  -1./2 * torch.dot(x,x)/(math.exp(y))
        case2matrix[self.n - 1, :(self.n - 1)].copy_(case2matrix[:(self.n-1),self.n-1])

        out[self.n-1,:,:].copy_(case2matrix)
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