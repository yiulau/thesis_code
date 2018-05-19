import torch.nn as nn
import torch
# assume applying the same prior to every parameter in the model
def sd_normal_prior_fun(model):
    # standard normal prior
    out = 0
    for param in model.list_var:
        out += -(param*param).sum() * 0.5
    return(out)

def horseshoe_prior_one_tau(model):
    out = 0
    for param in model.list_var:
        out += -(param*param).sum() * 0.5

    for param in model.list_tau:
        out +=

    for param in mo
    return(out)


def horseshoe_prior_tau_per_l(model):
    out = 0
    for param in model.list_var:
        out += -(param * param).sum() * 0.5

    for param in model.list_tau:
        out +=

    for param in mo
        return (out)

class prior_class(object):
    def __init__(self,V_nn_obj,prior_type):
        self.V_obj = V_nn_obj
        self.prior_type = prior_type
        self.prior_setup()
    def forward_var(self):
        if self.prior_type=="sd_normal":
            out = sd_normal_prior_fun(self.V_obj)
        elif self.prior_type=="hs":
            out = horseshoe_prior(self.V_obj,nu)
        elif self.prior_type=="hs_tau_per_l":
            out = horseshoe_prior(self.V_obj,nu)
        elif self.prior_type=="hs_ncp":
            pass
        elif self.prior_type=="hs_ncp_per_l":
            pass
        elif
        return(out)

    def prior_setup(self):
        if self.prior_type=="sd_normal":
            pass
        elif self.prior_type=="horseshoe_one_tau":
            for param_name,param in self.V_nn_obj.named_parameters():
                setattr(self.V_nn_obj,param_name+"lam",nn.Parameter(torch.zeros(param.shape),requires_grad=True))

            setattr(self.V_nn_obj,"tau",nn.Paramter(torch.zeros(1),requires_grad=True))

        elif self.prior_type=="horseshoe_tau_per_l":
            for i in range(len(self.V_nn_obj.named_params_by_layer)):
                for param_name,param in self.V_nn_obj.named_params_by_layer[i]:
                    setattr(self.V_nn_obj, param_name + "lam",
                            nn.Parameter(torch.zeros(param.shape), requires_grad=True))

                setattr(self.V_nn_obj, "tau"+"l{}".format(i), nn.Paramter(torch.zeros(1), requires_grad=True))

