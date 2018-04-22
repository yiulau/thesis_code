from abstract_class_V import T
import torch
class T_unit_e(T):
    def __init__(self):
        super(T_unit_e, self).__init__()
        return()

    def evaluate_float(self):
        output = 0
        for i in range(len(self.p)):
            output += (self.p[i] * self.p[i]).sum() * 0.5
        return(output)

    def dp(self,flattened_tensor):
        out = flattened_tensor
        return(out)
    def dtaudp(self):
        return (self.p)

    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self):
        for i in range(len(self.store_momentum)):
            self.store_momentum[i].copy_(torch.randn(self.list_shapes[i]))
        return(self.store_momentum)