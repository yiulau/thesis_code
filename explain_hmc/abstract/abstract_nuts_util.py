from torch.autograd import Variable
import torch, numpy, math
from explicit.general_util import logsumexp, stable_sum
from time_diagnostics import time_diagnositcs

def abstract_NUTS(q_init,epsilon,Ham,leapfrog,max_tdepth):
    # input and output are point objects
    Ham.diagnostics = time_diagnositcs()
    p_init = Ham.T.generate_momentum(q_init)

    q_left = q_init.point_clone()
    q_right =q_init.point_clone()
    p_left = p_init.point_clone()
    p_right = p_init.point_clone()
    j = 0
    num_div = 0
    q_prop = q_init.point_clone()
    log_w = -Ham.evaluate(q_init,p_init)
    H_0 = -log_w
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, p_prime,s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q_left, p_left, -1, j, epsilon, leapfrog, Ham,H_0
                                                                            )
        else:
            _, _, q_right, p_right, q_prime, p_prime, s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q_right, p_right, 1, j, epsilon, leapfrog, Ham,H_0
                                                                              )
        if s_prime:
            accept_rate = math.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()

        log_w = logsumexp(log_w,log_w_prime)
        s = s_prime and abstract_NUTS_criterion(q_left,q_right,p_left,p_right)
        j = j + 1
        s = s and (j<max_tdepth)
        num_div += num_div_prime
        Ham.diagnostics.update_time()
    return(q_prop,p_prop,p_init,j,num_div)
def abstract_GNUTS(q_init,epsilon,Ham,leapfrog,max_tdepth):
    # sum_p should be a tensor instead of variable
    p_sharp_fun = Ham.p_sharp
    Ham.diagnostics = time_diagnositcs()
    p_init = Ham.T.generate_momentum(q_init)
    q_left = q_init.point_clone()
    q_right = q_init.point_clone()
    p_left = p_init.point_clone()
    p_right = p_init.point_clone()
    p_sleft = p_sharp_fun(q_init, p_init).point_clone()
    p_sright = p_sharp_fun(q_init, p_init).point_clone()
    j = 0
    num_div = 0
    q_prop = q_init.point_clone()
    log_w = -Ham.evaluate(q_init,p_init)
    H_0 = -log_w
    sum_p = p_init.flattened_tensor.clone()
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, p_prime,s_prime, log_w_prime,sum_dp,num_div_prime = abstract_BuildTree_gnuts(q_left, p_left, -1, j, epsilon, leapfrog, Ham,
                                                                            p_sharp_fun,H_0)
        else:
            _, _, q_right, p_right, q_prime, p_prime,s_prime, log_w_prime, sum_dp,num_div_prime = abstract_BuildTree_gnuts(q_right, p_right, 1, j, epsilon, leapfrog, Ham,
                                                                              p_sharp_fun,H_0)
        if s_prime:
            accept_rate = math.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()
        log_w = logsumexp(log_w,log_w_prime)
        sum_p += sum_dp
        p_sleft = p_sharp_fun(q_left, p_left)
        p_sright = p_sharp_fun(q_right, p_right)
        s = s_prime and abstract_gen_NUTS_criterion(p_sleft,p_sright,sum_p)
        j = j + 1
        s = s and (j<max_tdepth)
        num_div +=num_div_prime
    Ham.diagnostics.update_time()
    return(q_prop,p_prop,p_init,j,num_div)
def abstract_NUTS_xhmc(q_init,epsilon,Ham,leapfrog,max_tdepth,xhmc_delta):
    dG_dt = Ham.dG_dt
    Ham.diagnostics = time_diagnositcs()
    p_init = Ham.T.generate_momentum(q_init)
    q_left = q_init.point_clone()
    q_right = q_init.point_clone()
    p_left = p_init.point_clone()
    p_right = p_init.point_clone()
    j = 0
    num_div = 0
    q_prop = q_init.point_clone()
    log_w = -Ham.evaluate(q_init,p_init)
    H_0 = -log_w
    ave = dG_dt(q_init, p_init)
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime,p_prime, s_prime, log_w_prime,ave_dp,num_div_prime = abstract_BuildTree_nuts_xhmc(q_left, p_left, -1, j, epsilon, leapfrog, Ham,
                                                                            dG_dt,xhmc_delta,H_0)
        else:
            _, _, q_right, p_right,p_prime, q_prime, s_prime, log_w_prime,ave_dp,num_div_prime = abstract_BuildTree_nuts_xhmc(q_right, p_right, 1, j, epsilon, leapfrog, Ham,
                                                                              dG_dt,xhmc_delta,H_0)
        if s_prime:
            accept_rate = math.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()
        oo = stable_sum(ave, log_w, ave_dp, log_w_prime)
        ave = oo[0]
        log_w = oo[1]
        s = s_prime and abstract_xhmc_criterion(ave,xhmc_delta,math.pow(2,j))
        j = j + 1
        s = s and (j<max_tdepth)
        num_div +=num_div_prime
    Ham.diagnostics.update_time()
    return(q_prop,p_prop,p_init,j,num_div)
def abstract_BuildTree_nuts(q,p,v,j,epsilon,leapfrog,Ham,H_0):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,Ham)
        log_w_prime = -Ham.evaluate(q_prime, p_prime)
        H_cur = -log_w_prime

        if (abs(H_cur - H_0) < 1000):
            # continue_divergence
            # boolean True if there's no divergence.
            continue_divergence = True
            num_div = 0
        else:
            continue_divergence = False
            num_div = 1
        return q_prime, p_prime, q_prime, p_prime, q_prime, p_prime, continue_divergence, log_w_prime, num_div
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime, p_prime,s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q, p, v, j - 1, epsilon, leapfrog, Ham,H_0)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,p_dprime,s_dprime,log_w_dprime,num_div_dprime = abstract_BuildTree_nuts(q_left,p_left,v,j-1,epsilon,leapfrog,Ham,H_0)
            else:
                _, _, q_right, p_right, q_dprime,p_dprime, s_dprime, log_w_dprime,num_div_dprime = abstract_BuildTree_nuts(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, Ham,H_0)
            accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accept_rate:
                q_prime = q_dprime.point_clone()
                p_prime = p_dprime.point_clone()
            s_prime = s_dprime and abstract_NUTS_criterion(q_left,q_right,p_left,p_right)
            num_div_prime +=num_div_dprime
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, p_prime, s_prime, log_w_prime,num_div_prime

def abstract_BuildTree_gnuts(q,p,v,j,epsilon,leapfrog,Ham,p_sharp_fun,H_0):

    #p_sharp_fun(q,p) takes tensor returns tensor

    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,Ham)
        log_w_prime = -Ham.evaluate(q_prime, p_prime)
        H_cur = -log_w_prime
        if (abs(H_cur - H_0) < 1000):
            continue_divergence = True
            num_div = 0
        else:
            continue_divergence = False
            num_div = 1
        return q_prime, p_prime, q_prime, p_prime, q_prime,p_prime, continue_divergence, log_w_prime,p_prime.flattened_tensor,num_div
    else:
        # first half of subtree
        sum_p = torch.zeros(len(p.flattened_tensor))
        q_left, p_left, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime,temp_sum_p,num_div_prime = abstract_BuildTree_gnuts(q, p, v, j - 1,
                                                                                                     epsilon, leapfrog,
                                                                                                     Ham,p_sharp_fun,H_0)
        sum_p += temp_sum_p
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,p_dprime,s_dprime,log_w_dprime,sum_dp,num_div_dprime = abstract_BuildTree_gnuts(q_left,p_left,v,j-1,epsilon,
                                                                                          leapfrog,Ham,
                                                                                          p_sharp_fun,H_0)
            else:
                _, _, q_right, p_right, q_dprime, p_dprime,s_dprime, log_w_dprime,sum_dp,num_div_dprime = abstract_BuildTree_gnuts(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, Ham,p_sharp_fun,H_0)
            accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accept_rate:
                q_prime = q_dprime.point_clone()
                p_prime = p_dprime.point_clone()
            sum_p += sum_dp
            num_div_prime +=num_div_dprime
            p_sleft = p_sharp_fun(q_left,p_left)
            p_sright = p_sharp_fun(q_right,p_right)
            s_prime = s_dprime and abstract_gen_NUTS_criterion(p_sleft, p_sright, sum_p)
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime,sum_p,num_div_prime
def abstract_BuildTree_nuts_xhmc(q,p,v,j,epsilon,leapfrog,Ham,dG_dt,xhmc_delta,H_0):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,Ham)
        log_w_prime = -Ham.evaluate(q_prime, p_prime)
        H_cur = -log_w_prime
        if(abs(H_cur-H_0)<1000):
            continue_divergence = True
            num_div = 0
        else:
            continue_divergence = False
            num_div = 1
        ave = dG_dt(q, p)
        return q_prime, p_prime, q_prime, p_prime, q_prime,p_prime, continue_divergence, log_w_prime, ave, num_div
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime, ave_prime,num_div_prime = abstract_BuildTree_nuts_xhmc(q, p, v, j - 1,
                                                                                                         epsilon, leapfrog,
                                                                                                         Ham,dG_dt,xhmc_delta,H_0)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,p_dprime,s_dprime,log_w_dprime, ave_dprime,num_div_dprime = abstract_BuildTree_nuts_xhmc(q_left,p_left,v,j-1,
                                                                                              epsilon,leapfrog,
                                                                                              Ham,dG_dt,xhmc_delta,H_0)
            else:
                _, _, q_right, p_right, q_dprime,p_dprime, s_dprime, log_w_dprime, ave_dprime,num_div_dprime = abstract_BuildTree_nuts_xhmc(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, Ham,dG_dt,xhmc_delta,H_0)
            accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accept_rate:
                q_prime = q_dprime.point_clone()
                p_prime = p_dprime.point_clone()
            oo_ = stable_sum(ave_prime, log_w_prime, ave_dprime, log_w_prime)
            ave_prime = oo_[0]
            log_w_prime = oo_[1]
            num_div_prime += num_div_dprime
            s_prime = s_dprime and abstract_xhmc_criterion(ave_prime,xhmc_delta,math.pow(2,j))

        return q_left, p_left, q_right, p_right, q_prime, p_prime,s_prime,log_w_prime,ave_prime,num_div_prime

def abstract_NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue doubling the trajectory
    # False = stop
    o = (torch.dot(p_right.flattened_tensor,q_right.flattened_tensor-q_left.flattened_tensor) >=0) or \
        (torch.dot(p_left.flattened_tensor,q_right.flattened_tensor-q_left.flattened_tensor) >=0)
    return(o)

def abstract_gen_NUTS_criterion(p_sleft,p_sright,p_sum):
    # p_sum should be a tensor
    # True = continue doubling the trajectory
    # False = stop
    o = (torch.dot(p_sleft.flattened_tensor,p_sum) >= 0) or \
        (torch.dot(p_sright.flattened_tensor,p_sum) >= 0)
    return(o)

def abstract_xhmc_criterion(ave,xhmc_delta,traj_len):
    o = abs(ave)/traj_len > xhmc_delta
    return(o)
