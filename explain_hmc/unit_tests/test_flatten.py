from abstract.abstract_class_V import V
import torch
import torch.nn as nn

class V_test_flatten(V):
    def V_setup(self):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.beta1 = nn.Parameter(torch.FloatTensor([[2.,4.],[3,5.]]),requires_grad=True)
        self.beta2 = nn.Parameter(torch.FloatTensor([6.,1]),requires_grad=True)
        return()
    def forward(self):

        out = (self.beta1*self.beta1*self.beta1).sum()*(self.beta2*self.beta2*self.beta2).sum()
        return(out)

class V_test_flatten2(V):
    def V_setup(self):
        self.explicit_gradient = False
        self.need_higherorderderiv = True
        self.beta = nn.Parameter(torch.FloatTensor([2.,4.,3,5.,6.,1]),requires_grad=True)
        return()
    def forward(self):
        beta1 = self.beta[0:4]
        beta2 = self.beta[4:]

        out = (beta1*beta1*beta1).sum()*(beta2*beta2*beta2).sum()
        return(out)




v_obj_separated = V_test_flatten()
v_obj_flattened = V_test_flatten2()

#print(v_obj_separated.need_flatten)

#print(v_obj_flattened.need_flatten)


# same output
#print(v_obj_separated.forward())
#print(v_obj_flattened.forward())

############################################################################################
# same gradient
# test 1
#print(v_obj_flattened.getdV())
#v_obj_flattened.getdV()

#print(v_obj_flattened.gradient_tensor)

#v_obj_separated.getdV()
#print(v_obj_separated.getdV())
print(v_obj_separated.gradient_tensor)
diff_grad = v_obj_flattened.gradient_tensor - v_obj_separated.gradient_tensor
print((diff_grad*diff_grad).sum())

#test2###################################################################################
flat_grad = v_obj_flattened.getdV_tensor()
print(flat_grad)
separated_grad = v_obj_separated.getdV_tensor()
print(separated_grad)
diff_grad = separated_grad - flat_grad
print((diff_grad*diff_grad).sum())
#exit()
############################################################################################
# same Hessian
# test1
flattenout = v_obj_flattened.getH()
separatedout = v_obj_separated.getH()

print(v_obj_flattened.Hessian_tensor)
print(v_obj_separated.Hessian_tensor)
diff_H = v_obj_separated.Hessian_tensor - v_obj_separated.Hessian_tensor
print((diff_H*diff_H).sum())

############################################################################################
# test2
_,flat_H_tensor = v_obj_flattened.getH_tensor()
print(flat_H_tensor)
_,separated_H_tensor = v_obj_separated.getH_tensor()
print(separated_H_tensor)
diff_H_tensor = separated_H_tensor - flat_H_tensor
print((diff_H_tensor*diff_H_tensor).sum())

##############################################################################################
# same dH
#test 1
flattenout2 = v_obj_flattened.getdH()
separatedout2 = v_obj_separated.getdH()
print(v_obj_flattened.dH_tensor)
print(v_obj_separated.dH_tensor)
diff_dH = v_obj_separated.dH_tensor - v_obj_separated.dH_tensor
print((diff_dH*diff_dH).sum())


# test 2 #######################################################################################

_,_,flat_dH_tensor = v_obj_flattened.getdH_tensor()
print(flat_dH_tensor)
_,_,separated_dH_tensor = v_obj_separated.getdH_tensor()
print(separated_dH_tensor)
diff_dH_tensor = separated_dH_tensor - flat_dH_tensor
print((diff_dH_tensor*diff_dH_tensor).sum())



################################################################################################
# same diagH
# test 1
_,flattenout2 = v_obj_flattened.getdiagH_tensor()
_,separatedout2 = v_obj_separated.getdiagH_tensor()
print(flattenout2)
print(separatedout2)
print(v_obj_flattened.diagH_tensor)
print(v_obj_separated.diagH_tensor)

diff_diagH = v_obj_separated.diagH_tensor - v_obj_separated.diagH_tensor
diff_diagH2 = flattenout2 - separatedout2
print((diff_diagH*diff_diagH).sum())
print((diff_diagH2*diff_diagH2).sum())
####################################################################################################

# same graddiagH

_,_,flat_graddiagH = v_obj_flattened.get_graddiagH()
_,_,separated_graddiagH = v_obj_separated.get_graddiagH()

print(flat_graddiagH)
print(separated_graddiagH)

diff_graddiagH = flat_graddiagH - separated_graddiagH
print((diff_graddiagH*diff_graddiagH).sum())
