from abstract_class_V_deprecated import V
import torch,math
import torch.nn as nn

precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)


class V_funnel(V):
    def V_setup(self):
        self.n = 10
        self.explicit_gradient = False
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
        p_y = -y*y * 1/9.* 0.5
        p_x = -(torch.dot(x,x))/(torch.exp(y*0.5)*torch.exp(y*0.5)) * 0.5
        prob = p_y + p_x
        out = -prob
        return(out)

    def load_explicit_gradient(self):
        out = torch.zeros(self.n)
        x = self.beta[:(self.n - 1)].data
        y = self.beta[self.n - 1].data
        out[:(self.n - 1)] = x/(math.exp(y/2)*math.exp(y/2))
        out[self.n-1] = float(y)/9 - torch.dot(x,x)/(2*math.exp(y/2)*math.exp(y/2))
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.n,self.n)
        x = self.beta[:(self.n - 1)].data
        y = self.beta[self.n - 1].data
        out[:(self.n - 1),:(self.n - 1)]  = 1/(math.exp(y/2)*math.exp(y/2)) * torch.eye(self.n-1,self.n-1)
        out[:(self.n-1),self.n-1] = -x/(math.exp(y/2)*math.exp(y/2))
        out[self.n-1,self.n-1] = 1/9 + 1/2 * torch.dot(x,x)/(math.exp(y/2)*math.exp(y/2))
        out[self.n - 1,:(self.n - 1)].copy_(out[:(self.n-1),self.n-1])
        return(out)
    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.n,self.n,self.n)
        x = self.beta[:(self.n - 1)].data
        y = self.beta[self.n - 1].data

        #case 1

        for i in range(self.n-1):
            out[i,self.n-1,i] = -1/( math.exp(y/2)*math.exp(y/2))
            out[i, i,self.n-1] = -1 / ( math.exp(y / 2) * math.exp(y / 2))
            out[i,self.n-1,self.n-1] = x[i]/ ( math.exp(y / 2) * math.exp(y / 2))
        #case 2
        case2matrix = torch.zeros(self.n,self.n)
        case2matrix[:(self.n - 1), :(self.n - 1)] =  -1/(math.exp(y/2)*math.exp(y/2)) * torch.eye(self.n-1,self.n-1)
        case2matrix[:(self.n-1),self.n-1] = x/(math.exp(y/2)*math.exp(y/2))
        case2matrix[self.n-1,self.n-1] =  -1./2 * torch.dot(x,x)/(math.exp(y/2)*math.exp(y/2))
        case2matrix[self.n - 1, :(self.n - 1)].copy_(case2matrix[:(self.n-1),self.n-1])

        out[self.n-1,:,:].copy_(case2matrix)
        return(out)