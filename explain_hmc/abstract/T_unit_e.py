from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
import torch
class T_unit_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_unit_e, self).__init__(linkedV)
        #return()

    def evaluate_scalar(self):
        output = 0
        for i in range(len(self.list_var)):
            output += (self.list_var[i].data * self.list_var[i].data).sum() * 0.5
        return(output)

    def dp(self,p_flattened_tensor):
        out = p_flattened_tensor
        return(out)
    def dtaudp(self):
        return (self.p)

    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self,q):
        out = point(None, self)
        out.flattened_tensor.copy_(torch.randn(self.dim))
        out.load_flatten()
        return(out)