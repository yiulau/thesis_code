from distributions.eightschool_ncp import V_eightschool_ncp
from unit_tests.finite_differences.finite_diff_funcs import finite_diff_grad,finite_diff_hessian,finite_diff_dH,compute_and_display_results

import torch

eightschool_ncp_object = V_eightschool_ncp()


eightschool_ncp_object.beta.data.copy_(torch.randn(10))
cur_beta = eightschool_ncp_object.beta.data.clone()

num_rep =10

compute_and_display_results(eightschool_ncp_object,3)

exit()
#print(eightschool_ncp_object.beta)
#exit()
#print(eightschool_ncp_object.forward())
####################################################################################################################
# gradient
# exact gradient

explicit_grad = eightschool_ncp_object.load_explicit_gradient()


#print(explicit_grad)

fin_diff_grad = finite_diff_grad(eightschool_ncp_object)


#print(fin_diff_grad)
l2norm_diff1stderiv=torch.dot(explicit_grad-fin_diff_grad,explicit_grad-fin_diff_grad)

#print("l2 norm difference between exact and finite diff for first derivs {} ".format(l2norm_diff1stderiv))



# finite_difference gradient


# autograd gradient
autograd_grad = eightschool_ncp_object.getdV().data
#print(autograd_grad)
l2norm_diff1stderiv_autograd=torch.dot(autograd_grad-fin_diff_grad,autograd_grad-fin_diff_grad)
#print("l2 norm difference between autograd and finite diff {} ".format(l2norm_diff1stderiv_autograd))

l2norm_diff1stderiv_autograd_explicit=torch.dot(autograd_grad-explicit_grad,autograd_grad-explicit_grad)
#print("l2 norm difference between autograd and exact diff {} ".format(l2norm_diff1stderiv_autograd_explicit))


#########################################################################################################
# hessian
# exact hessian

explicit_hessian = eightschool_ncp_object.load_explicit_H()

#print(explicit_hessian)

# finite difference hessian
fin_diff_hessian = finite_diff_hessian(eightschool_ncp_object)

#print(fin_diff_hessian)


l2norm_diff2ndderiv = ((explicit_hessian-fin_diff_hessian)*(explicit_hessian-fin_diff_hessian)).sum()
#print("l2 norm difference between exact and finite diff for the hessian {} ".format(l2norm_diff2ndderiv))

# autograd hessian
autograd_hessian = eightschool_ncp_object.getH().data

#print(autograd_hessian)
l2norm_diff2ndderiv_autograd = ((autograd_hessian-fin_diff_hessian)*(autograd_hessian-fin_diff_hessian)).sum()
#print("l2 norm difference between autograd and finite diff for the hessian {} ".format(l2norm_diff2ndderiv_autograd))

l2norm_diff2ndderiv_autograd_explicit = ((autograd_hessian-explicit_hessian)*(autograd_hessian-explicit_hessian)).sum()
#print("l2 norm difference between autograd and exact diff for the hessian {} ".format(l2norm_diff2ndderiv_autograd_explicit))

# dH
# exact dH

explicit_dH = eightschool_ncp_object.load_explicit_dH()




#print(explicit_dH)

# finite difference dH
fin_diff_dH = finite_diff_dH(eightschool_ncp_object)

autograd_dH = eightschool_ncp_object.getdH()
#############################################################################################################

l2norm_diff3rdderiv = ((explicit_dH-fin_diff_dH)*(explicit_dH-fin_diff_dH)).sum()
print("l2 norm difference between exact and finite diff for the dH {} ".format(l2norm_diff3rdderiv))


# autograd dh

autograd_dH = eightschool_ncp_object.getdH()
l2norm_diff3rdderiv_autograd = ((autograd_dH-fin_diff_dH)*(autograd_dH-fin_diff_dH)).sum()
print("l2 norm difference between autograd and finite diff for the dH {} ".format(l2norm_diff3rdderiv_autograd))


l2norm_diff3rdderiv_autograd_explicit = ((autograd_dH-explicit_dH)*(autograd_dH-explicit_dH)).sum()
print("l2 norm difference between autograd and exact diff for the dH {} ".format(l2norm_diff3rdderiv_autograd_explicit))

