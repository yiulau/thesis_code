from torch.autograd import Variable,grad
import torch, numpy, math
from general_util import logsumexp, logsumexp_torch


def NUTS(q_init,epsilon,H_fun,leapfrog,max_tdepth):
    p = Variable(torch.randn(len(q_init)),requires_grad=False)
    q_left = Variable(q_init.data.clone(),requires_grad=True)
    q_right = Variable(q_init.data.clone(),requires_grad=True)
    p_left = Variable(p.data.clone(),requires_grad=False)
    p_right = Variable(p.data.clone(),requires_grad=False)
    j = 0
    q_prop = Variable(q_init.data.clone(),requires_grad=True)
    log_w = -H_fun(q_init,p,return_float=True)
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, s_prime, log_w_prime = BuildTree_nuts(q_left, p_left, -1, j, epsilon, leapfrog, H_fun,
                                                                            )
        else:
            _, _, q_right, p_right, q_prime, s_prime, log_w_prime = BuildTree_nuts(q_right, p_right, 1, j, epsilon, leapfrog, H_fun,
                                                                              )
        if s_prime:
            accept_rate = math.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                q_prop.data = q_prime.data.clone()
        log_w = logsumexp(log_w,log_w_prime)
        s = s_prime and NUTS_criterion(q_left,q_right,p_left,p_right)
        j = j + 1
        s = s and (j<max_tdepth)
    return(q_prop,j)
def GNUTS(q_init,epsilon,H_fun,leapfrog,max_tdepth,p_sharp):
    # sum_p should be a tensor instead of variable
    p = Variable(torch.randn(len(q_init)),requires_grad=False)
    q_left = Variable(q_init.data.clone(),requires_grad=True)
    q_right = Variable(q_init.data.clone(),requires_grad=True)
    p_left = Variable(p.data.clone(),requires_grad=False)
    p_right = Variable(p.data.clone(),requires_grad=False)
    p_sleft = Variable(p_sharp(q_init, p).clone(), requires_grad=False)
    p_sright = Variable(p_sharp(q_init, p).clone(), requires_grad=False)
    j = 0
    q_prop = Variable(q_init.data.clone(),requires_grad=True)
    log_w = -H_fun(q_init,p,return_float=True)
    sum_p = p.data.clone()
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, s_prime, log_w_prime,sum_dp = BuildTree_gnuts(q_left, p_left, -1, j, epsilon, leapfrog, H_fun,
                                                                            )
        else:
            _, _, q_right, p_right, q_prime, s_prime, log_w_prime, sum_dp = BuildTree_gnuts(q_right, p_right, 1, j, epsilon, leapfrog, H_fun,
                                                                              )
        if s_prime:
            accept_rate = math.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                q_prop.data = q_prime.data.clone()
        log_w = logsumexp(log_w,log_w_prime)
        sum_p += sum_dp
        p_sleft = p_sharp(q_left, p_left)
        p_sright = p_sharp(q_right, p_right)
        s = s_prime and gen_NUTS_criterion(p_sleft,p_sright,sum_p)
        j = j + 1
        s = s and (j<max_tdepth)
    return(q_prop,j)
def BuildTree_nuts(q,p,v,j,epsilon,leapfrog,H_fun):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,H_fun)
        log_w_prime = -H_fun(q_prime, p_prime,return_float=True)
        return q_prime, p_prime, q_prime, p_prime, q_prime, True, log_w_prime
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime = BuildTree_nuts(q, p, v, j - 1, epsilon, leapfrog, H_fun)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,s_dprime,log_w_dprime = BuildTree_nuts(q_left,p_left,v,j-1,epsilon,leapfrog,H_fun)
            else:
                _, _, q_right, p_right, q_dprime, s_dprime, log_w_dprime = BuildTree_nuts(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, H_fun)
            accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accept_rate:
                q_prime.data = q_dprime.data.clone()
            s_prime = s_dprime and NUTS_criterion(q_left,q_right,p_left,p_right)
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime

def BuildTree_gnuts(q,p,v,j,epsilon,leapfrog,H_fun,p_sharp):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,H_fun)
        log_w_prime = -H_fun(q_prime, p_prime,return_float=True)
        return q_prime, p_prime, q_prime, p_prime, q_prime, True, log_w_prime,p_prime.data
    else:
        # first half of subtree
        sum_p = torch.zeros(len(p))
        q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime,temp_sum_p = BuildTree_gnuts(q, p, v, j - 1, epsilon, leapfrog, H_fun)
        sum_p += temp_sum_p
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,s_dprime,log_w_dprime,sum_dp = BuildTree_gnuts(q_left,p_left,v,j-1,epsilon,leapfrog,H_fun)
            else:
                _, _, q_right, p_right, q_dprime, s_dprime, log_w_dprime,sum_dp = BuildTree_gnuts(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, H_fun)
            accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accept_rate:
                q_prime.data = q_dprime.data.clone()
            sum_p += sum_dp
            p_sleft = p_sharp(q_left,p_left)
            p_sright = p_sharp(q_right,p_right)
            s_prime = s_dprime and gen_NUTS_criterion(p_sleft, p_sright, sum_p)
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime,sum_p

def NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue doubling the trajectory
    # False = stop
    o = (torch.dot(q_right.data-q_left.data,p_right.data) >=0) or \
        (torch.dot(q_right.data-q_left.data,p_left.data) >=0)
    return(o)

def gen_NUTS_criterion(p_left,p_right,p_sum):
    # p_sum should be a tensor
    # True = continue doubling the trajectory
    # False = stop
    o = (torch.dot(p_left.data,p_sum) >= 0) or \
        (torch.dot(p_right.data,p_sum) >= 0)
    return(o)