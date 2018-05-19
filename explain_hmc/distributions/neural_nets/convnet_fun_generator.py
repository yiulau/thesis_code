from abstract.abstract_class_V import V
import torch.nn as nn
from torch.autograd import Variable
import torch,numpy
def convnet_fun_generator(dataset,problem_type,num_units_list,activations_list,is_skip_connections,prior_dict):
    X = dataset["input"]
    target = dataset["target"]
    if problem_type == "regression":
        criterion = nn.MSELoss
        output_dim = 1
    elif problem_type == "classification":
        criterion = nn.NLLLoss
        output_dim = numpy.unique(target)
    class ConvNet(V):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7 * 7 * 32, output_dim)
            self.prepare_priors(prior_dict)

        def log_likelihood(self,X,y):
            if X == None:
                input_data = self.X
                target = self.target
            else:
                input_data = Variable(torch.from_numpy(X), requires_grad=False)
                target = Variable(torch.from_numpy(y), requires_grad=False)
            out = self.layer1(input_data)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            loss = criterion(out, target)
            return(loss)

        def log_prior(self):

            return()
        def forward(self, X,y):
            out_liklihood = self.log_likelihood(X,y)
            out_prior = self.log_prior()
            return(loss)
        def p_y_given_theta(self, observed_datapoint, posterior_param_point):
            self.load_point(posterior_param_point)
            out = self.forward(X=observed_datapoint["input"],y=observed_datapoint["target"])
            out = torch.exp(-out * 0.5)
            return (out.data[0])

        def log_p_y_given_theta(self, observed_datapoint, posterior_point):
            self.load_point(posterior_point)
            out = -self.forward(X=observed_datapoint["input"],y=observed_datapoint["target"]) * 0.5
            return (out.data[0])

    return(ConvNet)