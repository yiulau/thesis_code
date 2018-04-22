from abstract_class_V import T
import torch
class T_diag_e(T):
    def __init__(self):
        super(T_diag_e, self).__init__()
        return()

    def evaluate_float(self,p=None):
        output = 0
        if p==None:
            for i in range(len(self.p)):
                output += (self.p[i] * self.metric.sd_list[i] * self.p[i] * self.metric.sd_list[i]) * 0.5
        else:
            for i in range(len(p)):
                output += (p[i] * self.metric.sd_list[i] * p[i] * self.metric.sd_list[i]) * 0.5
        return(output)
    def dp(self,flattened_tensor):
        out = self.metric._var_vec * flattened_tensor
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

    def generate_momentum(self):
        for i in range(len(self.store_momentum)):
            self.store_momentum[i].copy_(torch.randn(self.list_shapes[i]) * self.metric.sd_list[i])
        return(self.store_momentum)