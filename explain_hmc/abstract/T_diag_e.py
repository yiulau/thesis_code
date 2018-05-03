from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
import torch
class T_diag_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_diag_e, self).__init__(linkedV)

    def evaluate_scalar(self):
        output = 0
        for i in range(len(self.list_var)):
            output += (self.list_var[i].data * self.list_var[i].data*self.metric._var_list_tensor[i]).sum() * 0.5

        return(output)
    def dp(self,p_flattened_tensor):
        out = self.metric._var_vec * p_flattened_tensor
        return(out)
    def dtaudp(self,p=None):
        if p==None:
            for i in range(len(self.list_shapes)):
                self.gradient[i].copy_(self.metric_var_list[i] * self.p[i])
        else:
            for i in range(len(self.list_shapes)):
                self.gradient[i].copy_(self.metric_var_list[i] * p[i])
        return (self.gradient)

    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self,q):

        out = point(None, self)
        out.flattened_tensor.copy_(self.metric._sd_vec * torch.randn(self.dim))
        out.load_flatten()
        return(out)