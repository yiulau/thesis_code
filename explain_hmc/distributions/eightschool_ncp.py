from abstract.abstract_class_V import V
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy

precision_type = 'torch.DoubleTensor'
#precision_type = 'torch.FloatTensor'
torch.set_default_tensor_type(precision_type)
class V_eightschool_ncp(V):
    #def __init__(self):
    #    super(V_test_abstract, self).__init__()

    def V_setup(self):
        y = numpy.array([28, 8, -3, 7, -1, 1, 18, 12])
        sigma = numpy.array([15, 10, 16, 11, 9, 11, 10, 18])
        self.explicit_gradient = True
        self.need_higherorderderiv = True
        self.J = len(y)
        self.beta = nn.Parameter(torch.zeros(self.J+2),requires_grad=True)
        # beta[J+1] = log(tau) , beta[J]= mu
        # beta[:J] = theta_tilde
        self.y = Variable(torch.from_numpy(y), requires_grad=False).type(precision_type)
        self.sigma = Variable(torch.from_numpy(sigma), requires_grad=False).type(precision_type)

    def forward(self):

        tau = torch.exp(self.beta[self.J+1])
        mu = self.beta[self.J]
        theta_tilde = self.beta[:(self.J)]
        theta = theta_tilde *tau + mu
        logp_y = -((self.y - theta) * (self.y-theta) / (self.sigma*self.sigma)).sum() * 0.5
        logp_theta_tilde = -torch.dot(theta_tilde,theta_tilde) * 0.5
        logp_mu = -mu*mu * 0.5 * 1/25.
        logp_tau =  -torch.log((1+ (tau*tau/25))) +  self.beta[self.J+1]
        logposterior = logp_y + logp_theta_tilde + logp_mu + logp_tau
        out = -logposterior
        return(out)

    def output_desired_data(self):
        out = torch.zeros(len(self.beta))
        tau = torch.exp(self.beta[self.J+1])
        mu = self.beta[self.J ]
        theta_tilde = self.beta[:(self.J )]
        theta = theta_tilde * tau + mu
        out[:(self.J )].copy_(theta.data)
        out[self.J].copy_(mu.data)
        out[self.J+1].copy_(tau.data)
        return(out)
    def load_explicit_gradient(self):
        # write down explicit gradient
        out = torch.zeros(self.dim)
        tau = numpy.asscalar(torch.exp(self.beta[self.J + 1]).data.numpy())
        mu = numpy.asscalar(self.beta[self.J].data.numpy())
        theta_tilde = self.beta[:(self.J)].data
        theta = theta_tilde * tau + mu
        sigma = self.sigma.data
        y = self.y.data
        # dtheta_tilde
        out[:(self.J)].copy_((y - theta)*tau / (sigma*sigma)-theta_tilde)
        # dmu
        out[self.J] = (- mu / 25) + ((y-theta)/(sigma*sigma)).sum()
        # dtau'

        out[self.J + 1] = -2*tau*tau/(tau*tau+25) + ((y-theta)*theta_tilde*tau/(sigma*sigma)).sum() + 1
        out = -out
        return(out)
    def load_explicit_H(self):
        # write down explicit hessian
        out = torch.zeros(self.dim, self.dim)
        tau = numpy.asscalar(torch.exp(self.beta[self.J + 1]).data.numpy())
        mu = numpy.asscalar(self.beta[self.J].data.numpy())
        theta_tilde = self.beta[:(self.J)].data
        theta = theta_tilde * tau + mu
        sigma = self.sigma.data
        y = self.y.data
        # dtheta_tilde dtheta_tilde
        out[:self.J, :self.J].copy_(torch.eye(self.J, self.J) * (-tau*tau / (sigma * sigma) -1))
        # dtheta dmu
        out[:self.J, self.J] = -tau/(sigma*sigma)
        # dtheta_tilde dtau'
        out[:self.J, self.J + 1].copy_(((y-mu-2*tau*theta_tilde)*tau / (sigma * sigma)))
        # dmu dmu
        out[self.J, self.J] = -1/25 - (1/(sigma*sigma)).sum()
        # dmu dtau'
        out[self.J, self.J + 1] = - (theta_tilde/(sigma*sigma)).sum()*tau
        # dtau' dtau'
        out[self.J + 1, self.J + 1] =  -100*tau*tau/((tau*tau+25)**2) \
                                       + ((theta_tilde*tau*(y-mu-2*theta_tilde*tau))/(sigma*sigma)).sum()
        # dmu dtheta
        out[self.J, :self.J].copy_(out[:self.J, self.J])
        # dtau' dtheta_tilde
        out[(self.J + 1), :self.J].copy_(out[:self.J, self.J + 1])
        # dtau' dmu
        out[(self.J + 1), self.J] = out[self.J, self.J + 1]
        out = -out
        return(out)
    def load_explicit_dH(self):
        # write down explicit 3 rd derivatives
        # write down explicit 3 rd derivatives
        out = torch.zeros(self.dim, self.dim,self.dim)
        tau = numpy.asscalar(torch.exp(self.beta[self.J + 1]).data.numpy())
        mu = numpy.asscalar(self.beta[self.J].data.numpy())
        theta_tilde = self.beta[:(self.J)].data
        theta = theta_tilde * tau + mu
        sigma = self.sigma.data
        y = self.y.data

        # case 1
        # dH_i i = (0,..,self.J-1]
        #dtheta_tilde dtheta_tilde dtau'
        out[:self.J,:self.J,self.J+1] = torch.diag(-2*tau*tau/(sigma*sigma))
        # dtheta_tilde dmu dtau'
        out[:self.J,self.J,self.J+1] = -tau/(sigma*sigma)
        # dtheta_tilde dtau' dtau'
        out[:self.J,self.J+1,self.J+1] = tau*(y-mu-4*theta_tilde*tau)/(sigma*sigma)
        # fill in rest
        out[:self.J, self.J + 1,:self.J].copy_(out[:self.J,:self.J,self.J+1])
        out[:self.J, self.J + 1,self.J].copy_(out[:self.J,self.J,self.J+1])

        # case 2
        # dH_i i = self.J , beta_i = mu
        # dmu dtheta_tilde dtau'
        out[self.J,:self.J,self.J+1] = -tau/(sigma*sigma)
        # dmu dtau' dtau'
        out[self.J,self.J+1,self.J+1] = -(theta_tilde/(sigma*sigma)).sum()*tau
        # fill in
        out[self.J, self.J + 1,:self.J].copy_(out[self.J,:self.J,self.J+1])

        # case 3
        # dH_i i = self.J+1 , beta_i = tau'

        case_3matrix = torch.zeros(self.dim, self.dim)
        # dtau' dtheta_tilde dtheta_tilde
        case_3matrix[:self.J, :self.J].copy_(torch.diag(-2 * tau*tau/(sigma*sigma)))
        # dtau' dtheta_tilde dmu
        case_3matrix[:self.J, self.J] = (-tau)/(sigma*sigma)
        # dtau' dtheta_tilde dtau'
        case_3matrix[:self.J,self.J+1] = tau*(y-mu-4*tau*theta_tilde)/(sigma*sigma)
        # dtau' dmu dtau'
        case_3matrix[self.J,self.J+1] = -(theta_tilde/(sigma*sigma)).sum()*tau
        # dtau'dtau'dtau'
        case_3matrix[self.J + 1, self.J + 1] = 200*tau*tau*(tau*tau-25)/((tau*tau+25)**3) \
                                               + (theta_tilde*tau*(y-mu-4*tau*theta_tilde)/(sigma*sigma)).sum()

        # fill in
        case_3matrix[self.J,:self.J].copy_(case_3matrix[:self.J, self.J])
        case_3matrix[self.J + 1,:self.J].copy_(case_3matrix[:self.J,self.J+1])
        case_3matrix[self.J + 1,self.J]=case_3matrix[self.J,self.J+1]

        out[self.J + 1, :, :].copy_(case_3matrix)
        out = -out
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