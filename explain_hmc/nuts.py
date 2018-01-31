import numpy
import torch
def NUTS(q_init,epsilon):
    p = torch.randn(len(q_init))
    q_left = q_init.clone()
    q_right = q_init.clone()
    p_left = p.clone()
    p_right = p.clone()
    j = 0
    q_prop = q_init.clone()
    w = pi(q_init,p)
    s = True
    while s:
        v = (numpy.random.randn(1)>0)*2 -1
        if v < 0:
            q_left,p_left,_,_,q_prime,s_prime,w_prime = BuildTree(q_left,p_left,v,j,epsilon)
        else:
            _,_,q_right,p_right,q_prime,s_prime,w_prime = BuildTree(q_right,p_right,v,j,epsilon)
        if s_prime:
            accep_rate = min(1,w_prime/w)
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prop = q_prime
            w = w + w_prime
            s = s and NUTS_criterion(q_left,q_right,p_left,p_right)
            j = j + 1
    return(q_prop)

def BuildTree(q,p,v,j,epsilon):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon)
        w_prime = pi(q_prime,p_prime)
        return q_prime,p_prime,q_prime,p_prime,q_prime,True,w_prime
    else:
        q_left,p_left,q_right,p_right,q_prime,s_prime,w_prime = BuildTree(q,p,v,j-1,epsilon)
        if s_prime:
            if v ==-1:
                q_left,p_left,_,_,q_dprime,s_dprime,w_dprime = BuildTree(q_left,p_left,v,j-1,epsilon)
            else:
                _,_,q_right,p_right,q_dprime,s_dprime,w_dprime = BuildTree(q_right,p_right,v,j-1,epsilon)
            accep_rate = min(1,w_dprime/(w_prime+w_dprime))
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prime = q_dprime.clone()
            s_prime = s_dprime and NUTS_criterion(q_left,q_right,p_left,p_right)
            w_prime = w_prime + w_dprime
        return q_left,p_left,p_left,p_right,q_prime,s_prime
