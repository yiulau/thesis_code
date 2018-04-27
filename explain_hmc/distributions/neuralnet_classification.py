from abstract_class_V import V
import torch
import torch.nn as nn
from torch.autograd import Variable, Function


# 2 layer regression network


class V_test_abstract(V):
    def __init__(self):
        super(V_test_abstract, self).__init__()

    def V_setup(self,X,target,lay1_num,lay2_num):
        self.explicit_gradient = False
        self.need_higherorderderiv = False

        self.lay1_num = lay1_num
        self.lay2_num = lay2_num
        self.X = Variable(torch.from_numpy(X),requires_grad=False)
        self.layer1 = nn.Linear(self.X.shape[1],lay1_num)
        self.layer2 = nn.Linear(self.lay1_num,self.lay2_num)
        self.target = Variable(torch.from_numpy(target),requires_grad=False)

        return()

    def forward(self,X=None,y=None):
        if X==None:
            input_data = self.X
            target = self.target
        else:
            input_data = Variable(torch.from_numpy(X),requires_grad=False)
            target = Variable(torch.from_numpy(y),requires_grad=False)

        out = Function.relu(self.layer1(input_data))
        out = self.layer2(out)
        criterion = nn.NLLLoss
        loss = criterion(out, target)
        return(loss)

    def p_y_given_theta(self,observed_point,posterior_point):
        self.load_point(posterior_point)
        out = self.forward(input=observed_point)
        out = torch.exp(-out)
        return(out.data[0])
    def log_p_y_given_theta(self,observed_point,posterior_point):
        self.load_point(posterior_point)
        out = -self.forward(input=observed_point)
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