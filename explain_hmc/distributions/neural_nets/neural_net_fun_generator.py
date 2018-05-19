import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from abstract.abstract_class_V import V
import numpy
def generate_nn_fun(dataset,problem_type,num_units_list,activations_list,is_skip_connections,prior_dict):
    # problem_type : one of ("classification","regression")
    # dataset : dict {"input":X,"target":y} should be numpy
    # minimum length 1 , each entry represent one hidden - layer
    # input layer >> hidden's >> output layer
    # num_units_list = [25,25] , len(num_units_list)=number of layers
    # num_units_list = activations_list = [relu,tanh] activation function objects
    # is_skip_connections : True or False, if True implements skip connections do nothing otherwise
    # prior function takes parameter and map to a Variable output
    X = dataset["input"]
    target = dataset["target"]
    if problem_type=="regression":
        criterion = nn.MSELoss
        output_dim = 1
    elif problem_type=="classification":
        criterion = nn.NLLLoss
        output_dim = numpy.unique(target)

    input_dim = X.shape[1]

    class out_class(V):
        def __init__(self):
            super(out_class, self).__init__()

        def V_setup(self):
            self.explicit_gradient = False
            self.need_higherorderderiv = False
            #self.lay1_num = lay1_num
            #self.lay2_num = lay2_num
            self.X = Variable(torch.from_numpy(X), requires_grad=False)
            #self.layer1 = nn.Linear(self.X.shape[1], lay1_num)
            #self.layer2 = nn.Linear(self.lay1_num, self.lay2_num)
            self.target = Variable(torch.from_numpy(target),requires_grad=False)
            units_list = [input_dim] + num_units_list
            units_list.append(output_dim)
            self.layer_objs_list = []
            for i in range(len(units_list)-1):
                obj = nn.Linear(units_list[i], units_list[i + 1])
                setattr(self,name="layer"+"{}".format(i+1),value=obj)
                self.layer_objs_list.append(obj)
            self.prepare_prior(prior_dict)


            return ()

        def log_likelihood(self,X,y):
            if X == None:
                input_data = self.X
                target = self.target
            else:
                input_data = Variable(torch.from_numpy(X), requires_grad=False)
                target = Variable(torch.from_numpy(y), requires_grad=False)

            out = activations_list[0]((self.layer_objs_list[0](input_data)))
            for i in range(1,len(self.layer_objs_list)-1):
                activation = activations_list[i]
                if is_skip_connections:
                    out = activation(self.layer_objs_list[i](out))+ out
                else:
                    out = activation(self.layer_objs_list[i](out))
            out = self.layer_objs_list[len(self.layer_objs_list)-1](out)
            loss = criterion(out, target)
            return(loss)
        def log_prior(self):
            out = 0
            for param in self.list_var:
                out += -(param * param).sum() * 0.5
            return(out)
        def forward(self,X,y):
            #input_data = Variable(torch.from_numpy(X), requires_grad=False)
            #target = Variable(torch.from_numpy(y), requires_grad=False)
            out_log_likelihood = self.log_likelihood(X,y)
            out_log_prior = self.log_prior()
            output = out_log_likelihood + out_log_prior
            return (-output)

        def p_y_given_theta(self, observed_datapoint, posterior_param_point):
            self.load_point(posterior_param_point)
            out = self.forward(X=observed_datapoint["input"],y=observed_datapoint["target"])
            out = torch.exp(-out * 0.5)
            return (out.data[0])

        def log_p_y_given_theta(self, observed_datapoint, posterior_point):
            self.load_point(posterior_point)
            out = -self.forward(X=observed_datapoint["input"],y=observed_datapoint["target"]) * 0.5

            return (out.data[0])

        def output_desired_data(self):
            out = torch.zeros(len(self.beta))
            x_raw = self.beta[:(self.n - 1)]
            y_raw = self.beta[self.n - 1]
            y = y_raw * 3.
            x = x_raw * torch.exp(y * 0.5)
            out[:(self.n - 1)].copy_(x.data)
            out[self.n - 1].copy_(y.data)
            return (out)

        def load_explcit_gradient(self):
            return ()

