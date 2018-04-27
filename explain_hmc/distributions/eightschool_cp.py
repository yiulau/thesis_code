from abstract_class_V_deprecated import V
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

#torch.set_default_tensor_type('torch.DoubleTensor')
precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)
class V_eightschool_cp(V):
    #def __init__(self):
    #    super(V_eightschool_cp, self).__init__()

    def V_setup(self):
        y = numpy.array([28,8,-3,7,-1,1,18,12])
        sigma = numpy.array([15,10,16,11,9,11,10,18])
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.J = len(y)
        self.beta = nn.Parameter(torch.zeros(self.J+2),requires_grad=True)
        # beta[J+1] = log(tau) , beta[J]= mu
        self.y = Variable(torch.from_numpy(y),requires_grad=False).type(precision_type)
        self.sigma = Variable(torch.from_numpy(sigma),requires_grad=False).type(precision_type)
        return()

    def forward(self):
        #temp =self.y - self.beta[:(self.J-2)]
        tau = torch.exp(self.beta[self.J+1])
        mu = self.beta[self.J]
        theta = self.beta[:(self.J)]

        logp_y = -((self.y - theta) * (self.y-theta) / (self.sigma*self.sigma)).sum() * 0.5

        logp_theta = -(((theta-mu) * (theta-mu)) / (tau*tau)).sum() * 0.5
        logp_mu = -mu*mu * 1/(2*25.)
        logp_tau =  -torch.log((1+ (tau*tau/25))) + self.beta[self.J+1]
        logposterior = logp_y + logp_theta + logp_mu + logp_tau
        out = -logposterior
        return(out)

    def load_explicit_gradient(self):
        # write down explicit gradient
        out = torch.zeros(self.dim)
        tau = numpy.asscalar(torch.exp(self.beta[self.J+1]).data.numpy())
        mu = numpy.asscalar(self.beta[self.J].data.numpy())
        theta = self.beta[:(self.J)].data
        sigma = self.sigma.data

        # dtheta
        out[:(self.J)].copy_((self.y.data-theta)/(sigma*sigma) - (theta-mu)/(tau*tau))
        #dmu
        out[self.J] = (((theta-mu)/(tau*tau))).sum()-mu/25
        #dtau'
        temp = torch.dot(theta-mu,theta-mu)/(tau*tau)  -2*tau*tau/(tau*tau+25) + 1
        out[self.J+1] = temp
        out = -out
        return(out)

    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.dim,self.dim)
        tau = numpy.asscalar(torch.exp(self.beta[self.J + 1]).data.numpy())
        mu = numpy.asscalar(self.beta[self.J].data.numpy())
        theta = self.beta[:(self.J)].data
        sigma = self.sigma.data
        # dtheta dtheta
        out[:self.J,:self.J].copy_(torch.eye(self.J,self.J)*(-1/(sigma*sigma)-1/(tau*tau)))
        # dtheta dmu
        out[:self.J,self.J] = (1/(tau*tau))
        # dtheta dtau'
        out[:self.J,self.J+1].copy_((2*(theta-mu)/(tau*tau)))
        # dmu dmu
        out[self.J,self.J] = -self.J/(tau*tau) - 1/25.
        # dmu dtau'
        out[self.J,self.J+1] = -2 * (theta-mu).sum()/(tau*tau)
        # dtau' dtau'
        #out[self.J+1,self.J+1] = -2 * torch.dot(theta-mu,theta-mu)/(tau*tau) + 4*(tau*tau)/25 + 8*(tau*tau*tau*tau)/(25*25)
        out[self.J + 1, self.J + 1] = -2 * torch.dot(theta-mu,theta-mu)/(tau*tau) -100*tau*tau/((tau*tau+25)*(tau*tau+25))
        # dmu dtheta
        out[self.J, :self.J].copy_(out[:self.J,self.J])
        # dtau' dtheta
        out[(self.J + 1),:self.J].copy_(out[:self.J,self.J+1])

        # dtau' dmu
        out[(self.J + 1),self.J] = out[self.J,self.J+1]
        #print("yes{}".format(out))
        out = -out
        return(out)

    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.dim, self.dim,self.dim)
        tau = numpy.asscalar(torch.exp(self.beta[self.J + 1]).data.numpy())
        mu = numpy.asscalar(self.beta[self.J].data.numpy())
        theta = self.beta[:(self.J)].data
        sigma = self.sigma.data

        # case 1
        # dH_i i = (0,..,self.J-1]
        # excludes case_1matrix[self.dim-1,self.dim-1] because its the only entry that differs across i
        # fill in later
        case_1matrix = torch.zeros(self.dim,self.dim)
        #case_1matrix[:self.J,self.J+1] = 2/(tau*tau)
        case_1matrix[self.J,self.J+1] = -2/(tau*tau)
        #case_1matrix[self.J + 1,:self.J].copy_(case_1matrix[:self.J,self.J+1])
        case_1matrix[self.J+1, self.J] = case_1matrix[self.J,self.J+1]

        out[:self.J,:,:].copy_(case_1matrix.expand_as(out[:self.J,:,:]))
        # fill in out[i,self.dim-1,self.dim-1] for i = (0,..,self.J-1)
        out[:self.J,self.J+1,self.J+1].copy_(-4*(theta-mu)/(tau*tau))
        # fill in out[:self.J,:self.J,self.J+1]
        out[:self.J, :self.J, self.J + 1] = torch.eye(self.J,self.J)*(2 / (tau * tau))
        out[:self.J,self.J+1,:self.J].copy_(out[:self.J, :self.J, self.J + 1])
        # case 2
        # dH_i i = self.J , beta_i = mu
        case_2matrix = torch.zeros(self.dim, self.dim)
        case_2matrix[:self.J, self.J + 1] = -2 / (tau * tau)
        case_2matrix[self.J, self.J + 1] = 2*self.J / (tau * tau)
        case_2matrix[self.J+1, self.J + 1] = 4 * (theta-mu).sum()/(tau*tau)
        case_2matrix[self.J + 1,:self.J].copy_(case_2matrix[:self.J, self.J + 1])
        case_2matrix[self.J + 1, self.J] = case_2matrix[self.J, self.J + 1]

        out[self.J,:,:].copy_(case_2matrix)

        # case 3
        # dH_i i = self.J+1 , beta_i = tau'
        case_3matrix = torch.zeros(self.dim,self.dim)
        case_3matrix[:self.J,:self.J].copy_((2/(tau*tau))*torch.eye(self.J,self.J))
        case_3matrix[:self.J, self.J] = -2 / (tau * tau)
        case_3matrix[:self.J, self.J + 1] = -4 * (theta-mu)/(tau*tau)
        case_3matrix[self.J,self.J] = 2*self.J/(tau*tau)
        case_3matrix[self.J,self.J+1] = 4 * (theta-mu).sum()/(tau*tau)
        case_3matrix[self.J+1,self.J+1] = 4 * torch.dot(theta-mu,theta-mu)/(tau*tau) + 200*tau*tau*(tau*tau-25)/((tau*tau+25)**3)
        case_3matrix[self.J + 1,:self.J].copy_(case_3matrix[:self.J, self.J + 1])
        case_3matrix[self.J + 1,self.J] = case_3matrix[self.J, self.J + 1]
        case_3matrix[self.J,:self.J].copy_(case_3matrix[:self.J, self.J])

        out[self.J+1,:,:].copy_(case_3matrix)
        out = -out
        return(out)