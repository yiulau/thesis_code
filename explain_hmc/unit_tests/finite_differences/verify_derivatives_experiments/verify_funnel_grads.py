from distributions.funnel_cp import V_funnel
import torch
from unit_tests.finite_differences.finite_diff_funcs import compute_and_display_results
funnel_object = V_funnel()

funnel_object.beta.data.copy_(torch.randn(10))
compute_and_display_results(funnel_object,10)

exit()
cur_beta = funnel_object.beta.data.clone()


#print(funnel_object.beta)
#funnel_object.beta.data[0]=funnel_object.beta.data[0]+0.01

#print(funnel_object.forward())

# gradient
# exact gradient

explicit_grad = funnel_object.load_explicit_gradient()


#print(explicit_grad)
from unit_tests.finite_differences.finite_diff_funcs import finite_1stderiv
def finite_diff_grad():
    cur_beta = funnel_object.beta.data.clone()
    out = torch.zeros(10)
    for i in range(10):
        cur_vari = cur_beta[i]
        h = 1e-3
        def fun_wrapped(diffi):
            funnel_object.beta.data.copy_(cur_beta)
            funnel_object.beta.data[i] = funnel_object.beta.data[i] + diffi
            temp = funnel_object.forward().data[0]
            return(temp)
        out[i] = finite_1stderiv(fun_wrapped,h)
    return(out)
fin_diff_grad = finite_diff_grad()


print(fin_diff_grad)
l2norm_diff1stderiv=torch.dot(explicit_grad-fin_diff_grad,explicit_grad-fin_diff_grad)
print("l2 norm difference between exact and finite diff {} ".format(l2norm_diff1stderiv))



# autograd gradient
autograd_grad = funnel_object.getdV().data
#print(autograd_grad)
l2norm_diff1stderiv_autograd=torch.dot(autograd_grad-fin_diff_grad,autograd_grad-fin_diff_grad)
print("l2 norm difference between autograd and finite diff {} ".format(l2norm_diff1stderiv_autograd))

l2norm_diff1stderiv_autograd_explicit=torch.dot(autograd_grad-explicit_grad,autograd_grad-explicit_grad)
print("l2 norm difference between autograd and exact diff {} ".format(l2norm_diff1stderiv_autograd_explicit))


# hessian
# exact hessian

explicit_hessian = funnel_object.load_explicit_H()

#print(explicit_hessian)
# finite difference hessian

from unit_tests.finite_differences.finite_diff_funcs import finite_2ndderiv
def finite_diff_hessian():
    out = torch.zeros(10,10)
    for i in range(10):
        for j in range(10):
            h = 1e-2
            def fun_wrapped(diffi,diffj):
                funnel_object.beta.data.copy_(cur_beta)
                funnel_object.beta.data[i]=funnel_object.beta.data[i]+diffi
                funnel_object.beta.data[j]=funnel_object.beta.data[j]+diffj
                temp = funnel_object.forward().data[0]

                return(temp)
            #print(cur_vari)
            #print(cur_varj)
            #exit()
            out[i,j] = finite_2ndderiv(fun_wrapped,h)
    return(out)
fin_diff_hessian = finite_diff_hessian()


#print(fin_diff_hessian)

l2norm_diff2ndderiv = ((explicit_hessian-fin_diff_hessian)*(explicit_hessian-fin_diff_hessian)).sum()
print("l2 norm difference between exact and finite diff for the hessian {} ".format(l2norm_diff2ndderiv))

# autograd hessian

autograd_hessian = funnel_object.getH().data

#print(autograd_hessian)
l2norm_diff2ndderiv_autograd = ((autograd_hessian-fin_diff_hessian)*(autograd_hessian-fin_diff_hessian)).sum()
print("l2 norm difference between autograd and finite diff for the hessian {} ".format(l2norm_diff2ndderiv_autograd))

l2norm_diff2ndderiv_autograd_explicit = ((autograd_hessian-explicit_hessian)*(autograd_hessian-explicit_hessian)).sum()
print("l2 norm difference between autograd and exact diff for the hessian {} ".format(l2norm_diff2ndderiv_autograd_explicit))

# dH
# exact dH

explicit_dH = funnel_object.load_explicit_dH()

#print(explicit_dH)

# finite difference dH
from unit_tests.finite_differences.finite_diff_funcs import finite_3rdderiv
def finite_diff_dH():
    out = torch.zeros(10,10,10)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                h = 1e-3
                def fun_wrapped(diffi,diffj,diffk):
                    funnel_object.beta.data.copy_(cur_beta)
                    funnel_object.beta.data[i]=funnel_object.beta.data[i]+diffi
                    funnel_object.beta.data[j]=funnel_object.beta.data[j]+diffj
                    funnel_object.beta.data[k] = funnel_object.beta.data[k] + diffk
                    temp = funnel_object.forward().data[0]
                    funnel_object.beta.data.copy_(cur_beta)
                    return(temp)
                #print(cur_vari)
                #print(cur_varj)
                #exit()
                out[i,j,k] = finite_3rdderiv(fun_wrapped,h)
    return(out)
fin_diff_dH = finite_diff_dH()


import time
num_rep =10
time_total_explicit = 0
time_total_finite = 0
time_total_autograd = 0
store_diff_exact_finite = []
store_diff_autograd_finite = []
store_diff_autograd_exact = []
for i in range(num_rep):
    funnel_object.beta.data.copy_(torch.randn(10))
    cur_beta = funnel_object.beta.data.clone()
    time_temp = time.time()
    explicit_dH = funnel_object.load_explicit_dH()
    time_total_explicit += time.time()-time_temp
    time_temp = time.time()
    fin_diff_dH = finite_diff_dH()
    time_total_finite += time.time() - time_temp
    time_temp = time.time()
    autograd_dH = funnel_object.getdH()
    time_total_autograd += time.time()-time_temp

    l2norm_diff3rdderiv = ((explicit_dH-fin_diff_dH)*(explicit_dH-fin_diff_dH)).sum()
    store_diff_exact_finite.append(l2norm_diff3rdderiv)
    print("l2 norm difference between exact and finite diff for the dH {} ".format(l2norm_diff3rdderiv))

    l2norm_diff3rdderiv_autograd = ((autograd_dH - fin_diff_dH) * (autograd_dH - fin_diff_dH)).sum()
    store_diff_autograd_finite.append(l2norm_diff3rdderiv_autograd)
    print("l2 norm difference between autograd and finite diff for the dH {} ".format(l2norm_diff3rdderiv_autograd))

    l2norm_diff3rdderiv_autograd_explicit = ((autograd_dH - explicit_dH) * (autograd_dH - explicit_dH)).sum()
    store_diff_autograd_exact.append(l2norm_diff3rdderiv_autograd_explicit)
    print("l2 norm difference between autograd and exact diff for the dH {} ".format(
        l2norm_diff3rdderiv_autograd_explicit))

print("explicit time {}".format(time_total_explicit))
print("finite time {}".format(time_total_finite))
print("autograd time {}".format(time_total_autograd))

import numpy
print("mean exact-finite diff{}".format(numpy.mean(store_diff_exact_finite)))
print("mean autograd-finite diff{}".format(numpy.mean(store_diff_autograd_finite)))
print("mean autograd-exact diff{}".format(numpy.mean(store_diff_autograd_exact)))

exit()

#print(fin_diff_dH)

l2norm_diff3rdderiv = ((explicit_dH-fin_diff_dH)*(explicit_dH-fin_diff_dH)).sum()
print("l2 norm difference between exact and finite diff for the dH {} ".format(l2norm_diff3rdderiv))

# autograd dh

autograd_dH = funnel_object.getdH()
l2norm_diff3rdderiv_autograd = ((autograd_dH-fin_diff_dH)*(autograd_dH-fin_diff_dH)).sum()
print("l2 norm difference between autograd and finite diff for the dH {} ".format(l2norm_diff3rdderiv_autograd))


l2norm_diff3rdderiv_autograd_explicit = ((autograd_dH-explicit_dH)*(autograd_dH-explicit_dH)).sum()
print("l2 norm difference between autograd and exact diff for the dH {} ".format(l2norm_diff3rdderiv_autograd_explicit))

