from abstract_class_V import T
import torch
class T_dense_e(T):
    def __init__(self):
        super(T_dense_e, self).__init__()
        return()

    def evaluate_float(self,p=None):
        if p ==None:
            Lp = torch.mv(self.metric._flattened_covL, self.flattened_tensor)
        else:
            Lp = torch.mv(self.metric._flattened_covL, p.flattened_tensor)

        output = torch.dot(Lp, Lp)
        return(output)

    def dp(self,flattened_tensor):
        out = torch.mv(self.metric._flattened_cov,flattened_tensor)
        return(out)

    def dtaudp(self,p=None):
        if self.need_flatten:
            self.load_listp_to_flattened(self.p, self.flattened_p_tensor)
            temp = torch.mv(self.metric._flattened_cov, self.flattened_p_tensor)
            self.load_flattened_tenosr_to_target_list(self.gradient,self.flattened_p_tensor)
            return (self.gradient)
        else:
            self.load_flattened_tenosr_to_target_list(self.gradient,torch.mv(self.metric._flattened_cov, self.p[0]))
            return (self.gradient)

    def dp(self,p):
        out = torch.mv(self.metric._flattened_cov, self.p[0])
        return(out)
    def dtaudq(self):
        raise ValueError("should not call this function")

    def generate_momentum(self):
        self.flattened_graident.copy_(torch.mv(self.metric._flattened_covL, torch.randn(self.dim)))
        self.load_flattened_tenosr_to_target_list(self.flattened_gradient, self.store_momentum)
        return(self.store_momentum)