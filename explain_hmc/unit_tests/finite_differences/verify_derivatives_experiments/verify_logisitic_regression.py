from distributions.logistic_regressions.logistic_regression import V_logistic_regression
import torch
from unit_tests.finite_differences.finite_diff_funcs import compute_and_display_results

logistic_regression_object = V_logistic_regression()

dim = logistic_regression_object.dim
logistic_regression_object.beta.data.copy_(torch.randn(dim))


#cProfile.run("logistic_regression_object.getdH()[2]")
#cProfile.run("logistic_regression_object.getH()")
#exit()
compute_and_display_results(logistic_regression_object,10)

exit()
cur_beta = logistic_regression_object.beta.data.clone()


#print(logistic_regression_object.beta)

#funnel_object.beta.data[0]=funnel_object.beta.data[0]+0.01

print(logistic_regression_object.forward())

# gradient
# exact gradient

explicit_grad = logistic_regression_object.load_explicit_gradient()


#print(explicit_grad)

from unit_tests.finite_differences.finite_diff_funcs import finite_1stderiv
def finite_diff_grad():
    cur_beta = logistic_regression_object.beta.data.clone()
    out = torch.zeros(dim)
    for i in range(dim):
        cur_vari = cur_beta[i]
        h = 1e-5
        def fun_wrapped(diffi):
            logistic_regression_object.beta.data.copy_(cur_beta)
            logistic_regression_object.beta.data[i] = logistic_regression_object.beta.data[i] + diffi
            temp = logistic_regression_object.forward().data[0]
            return(temp)
        out[i] = finite_1stderiv(fun_wrapped,h)
    return(out)
fin_diff_grad = finite_diff_grad()


print(fin_diff_grad)
l2norm_diff1stderiv=torch.dot(explicit_grad-fin_diff_grad,explicit_grad-fin_diff_grad)

print("l2 norm difference between exact and finite diff for first derivs {} ".format(l2norm_diff1stderiv))



# finite_difference gradient


# autograd gradient
autograd_grad = logistic_regression_object.getdV().data
#print(autograd_grad)
l2norm_diff1stderiv_autograd=torch.dot(autograd_grad-fin_diff_grad,autograd_grad-fin_diff_grad)
print("l2 norm difference between autograd and finite diff {} ".format(l2norm_diff1stderiv_autograd))

l2norm_diff1stderiv_autograd_explicit=torch.dot(autograd_grad-explicit_grad,autograd_grad-explicit_grad)
print("l2 norm difference between autograd and exact diff {} ".format(l2norm_diff1stderiv_autograd_explicit))



# hessian
# exact hessian

explicit_hessian = logistic_regression_object.load_explicit_H()

#print(explicit_hessian)

# finite difference hessian

from unit_tests.finite_differences.finite_diff_funcs import finite_2ndderiv
def finite_diff_hessian():
    out = torch.zeros(dim,dim)
    for i in range(dim):
        for j in range(dim):
            h = 1e-5
            def fun_wrapped(diffi,diffj):
                logistic_regression_object.beta.data.copy_(cur_beta)
                logistic_regression_object.beta.data[i]=logistic_regression_object.beta.data[i]+diffi
                logistic_regression_object.beta.data[j]=logistic_regression_object.beta.data[j]+diffj
                temp = logistic_regression_object.forward().data[0]

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

autograd_hessian = logistic_regression_object.getH().data

#print(autograd_hessian)
l2norm_diff2ndderiv_autograd = ((autograd_hessian-fin_diff_hessian)*(autograd_hessian-fin_diff_hessian)).sum()
print("l2 norm difference between autograd and finite diff for the hessian {} ".format(l2norm_diff2ndderiv_autograd))

l2norm_diff2ndderiv_autograd_explicit = ((autograd_hessian-explicit_hessian)*(autograd_hessian-explicit_hessian)).sum()
print("l2 norm difference between autograd and exact diff for the hessian {} ".format(l2norm_diff2ndderiv_autograd_explicit))



# dH
# exact dH

explicit_dH = logistic_regression_object.load_explicit_dH()

#print(explicit_dH)

# finite difference dH
from unit_tests.finite_differences.finite_diff_funcs import finite_3rdderiv
def finite_diff_dH():
    out = torch.zeros(dim,dim,dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                h = 1e-3
                def fun_wrapped(diffi,diffj,diffk):
                    logistic_regression_object.beta.data.copy_(cur_beta)
                    logistic_regression_object.beta.data[i]=logistic_regression_object.beta.data[i]+diffi
                    logistic_regression_object.beta.data[j]=logistic_regression_object.beta.data[j]+diffj
                    logistic_regression_object.beta.data[k] = logistic_regression_object.beta.data[k] + diffk
                    temp = logistic_regression_object.forward().data[0]

                    return(temp)
                #print(cur_vari)
                #print(cur_varj)
                #exit()
                out[i,j,k] = finite_3rdderiv(fun_wrapped,h)
    return(out)
fin_diff_dH = finite_diff_dH()



l2norm_diff3rdderiv = ((explicit_dH-fin_diff_dH)*(explicit_dH-fin_diff_dH)).sum()
print("l2 norm difference between exact and finite diff for the dH {} ".format(l2norm_diff3rdderiv))


# autograd dh

autograd_dH = logistic_regression_object.getdH()
l2norm_diff3rdderiv_autograd = ((autograd_dH-fin_diff_dH)*(autograd_dH-fin_diff_dH)).sum()
print("l2 norm difference between autograd and finite diff for the dH {} ".format(l2norm_diff3rdderiv_autograd))


l2norm_diff3rdderiv_autograd_explicit = ((autograd_dH-explicit_dH)*(autograd_dH-explicit_dH)).sum()
print("l2 norm difference between autograd and exact diff for the dH {} ".format(l2norm_diff3rdderiv_autograd_explicit))

