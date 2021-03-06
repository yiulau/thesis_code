from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from abstract.abstract_class_Ham import Hamiltonian
from abstract.metric import metric
import torch
from abstract.abstract_leapfrog_ult_util import abstract_HMC_alt_ult

#seed=2
#torch.manual_seed(seed)
#input = torch.randn(2)*1e-3
#print(input)
#output = softabs_map(input,1e-3)
#print(output)
#exit()

#seed=1
#torch.manual_seed(seed)
#numpy.random.seed(seed)
vo = V_logistic_regression()

#vo = V_funnel()
#metrico = metric("softabs",vo,alpha=1e6)
#metrico = metric("softabs_diag",vo,alpha=1e6)
#metrico = metric("softabs_outer_product",vo,alpha=1e6)
#metrico = metric("diag_e",vo)
#metrico = metric("dense_e",vo)
metrico = metric("unit_e",vo)
#T_unit_e(metrico,vo)
#exit()
#to = T(metrico,vo)
Ho = Hamiltonian(vo,metrico)
epsilon = 0.1

qpoint_obj = Ho.V.q_point
q = qpoint_obj
#out = abstract_NUTS(q,epsilon,Ho,abstract_leapfrog_ult,5)
#out = abstract_GNUTS(q,epsilon,Ho,5)
#out = abstract_GNUTS(q,epsilon,Ho,abstract_leapfrog_ult,5)
#out = abstract_NUTS_xhmc(q,epsilon,Ho,abstract_leapfrog_ult,5,0.1)
#out = abstract_NUTS_xhmc(q,epsilon,Ho,generalized_leapfrog,5,0.1)
out = abstract_HMC_alt_ult(epsilon=0.01,L=10,init_q=qpoint_obj,Ham=Ho)
#out = rmhmc_step(qpoint_obj,0.01,10,Ho)
#out = abstract_HMC_alt_windowed(epsilon=0.01,L=10,current_q=qpoint_obj,leapfrog_window=abstract_leapfrog_window,Ham=Ho)

q_out = out[0]
#p_out = out[1]
print(q_out.flattened_tensor)
#print(p_out.flattened_tensor)
#print(out[2].flattened_tensor)
print(out[3:])
exit()
#vo.beta.data = torch.randn(2)
print(vo.list_var[0].data)
qpoint_obj = Ho.V.q_point
print(qpoint_obj.list_var[0].data)





out = abstract_HMC_alt_ult(epsilon=0.1,L=10,current_q=qpoint_obj,leapfrog=abstract_HMC_alt_ult,Ham=Ho)

q_out = out[0]

print(q_out.flattened_tensor)
exit()
out_grad = Ho.V.getdV().data

out_gradtensor = Ho.V.getdV_tensor()


#print(out_grad)
#print(out_grad-out_gradtensor)

_,out_H = Ho.V.getH()

out_H = out_H.data
_,out_Htensor = Ho.V.getH_tensor()


#print(out_H)
#print(out_Htensor)

#print(out_H-out_Htensor)

_,_,out_dH = Ho.V.getdH()

_,_,out_dHtensor = Ho.V.getdH_tensor()

#print(out_dH-out_dHtensor)


# test point

qpoint_obj = Ho.V.point_generator()


# test one step
from abstract.abstract_leapfrog_ult_util import abstract_HMC_alt_ult

out = abstract_HMC_alt_ult(epsilon=0.1,L=10,current_q=qpoint_obj,leapfrog=abstract_HMC_alt_ult,Ham=Ho)

q_out = out[0]


print(q_out.flattened_tensor)
exit()
out = vo.flattened_tensor

out_grad = vo.gradient_tensor

print(out)
print(out_grad)
out2 = vo.list_tensor

print(hex(id(out2[0])))

print(hex(id(out)))


vo.flattened_tensor[0:2] = torch.randn(2)

print(out)

print(out2[0])
