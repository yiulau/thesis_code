import math,numpy
from explicit.general_util import logsumexp

# all functions modify (q,p)
def abstract_leapfrog_ult(q,p,epsilon,V,T):
    # Input:
    # q, p point objects
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)
    p.flattened_tensor -= V.dq(q.flattened_tensor) * 0.5 * epsilon
    q.flattened_tensor += T.dp(p.flattened_tensor) * epsilon
    p.flattened_tensor -= V.dq(q.flattened_tensor) * 0.5 * epsilon
    p.load_flatten()
    q.load_flatten()

    return(q,p)

def abstract_leapfrog_window(q,p,epsilon,V,T,logw_old,qprop_old,pprop_old):
    # Input: q,p current (q,p) state in trajecory
    # q,p point objects
    # qprop_old, pprop_old current proposed states in trajectory
    # logw_old = -H(qprop_old,pprop_old,return_float=True)
    q,p = abstract_leapfrog_ult(q,p,epsilon,V,T)
    logw_prop = -V.evaluate_float(q) - T.evaluate_float(p)
    accep_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
    u = numpy.random.rand(1)[0]
    if u < accep_rate:
        qprop = q
        pprop = p
    else:
        qprop = qprop_old
        pprop = pprop_old
        logw_prop = logw_old
    return(q,p,qprop,pprop,logw_prop,accep_rate)

def abstract_HMC_alt_ult(epsilon, L, current_q, leapfrog, V,T,generate_momentum,evol_t=None):
    # Input:
    # current_q Pytorch Variable
    # H_fun(q,p,return_float) returns Pytorch Variable or float
    # generate_momentum(q) returns pytorch variable
    # Output:
    # accept_rate: float - probability of acceptance
    # accepted: Boolean - True if proposal is accepted, False otherwise
    # divergent: Boolean - True if the end of the trajectory results in a divergent transition
    # return_q  pytorch Variable (! not tensor)
    #q = Variable(current_q.data.clone(),requires_grad=True)
    if not L==None and not evol_t==None:
        raise ValueError("L contradicts with evol_t")
    p = T.generate_momentum()
    q = current_q
    current_H = V(q) + T(p)
    for _ in range(L):
        q,p = leapfrog(q, p, epsilon,V,T)

    proposed_H = V(q) + T(p)
    diff = current_H - proposed_H
    if (abs(diff) > 1000):
        return_q = current_q
        return_H = current_H
        accept_rate = 0
        accepted = False
        divergent = True
    else:
        accept_rate = math.exp(min(0,diff))
        divergent = False
        if (numpy.random.random(1) < accept_rate):
            accepted = True
            return_q = q
            return_H = proposed_H
        else:
            accepted = False
            return_q = current_q
            return_H = current_H
    return(return_q,return_H,accepted,accept_rate,divergent)

def abstract_HMC_alt_windowed(epsilon, L, current_q, leapfrog_window, V,T,evol_t=None):
    if not L==None and not evol_t==None:
        raise ValueError("L contradicts with evol_t")
    p = T.generate_momentum()
    q = current_q
    logw_prop = -V(q) - T(p)
    q_prop = q.point_clone()
    p_prop = p.point_clone()
    accep_rate_sum = 0
    for _ in range(L):
        o = leapfrog_window(q, p, epsilon, V,T,logw_prop,q_prop,p_prop)
        q.data, p.data = o[0].data.clone(), o[1].data.clone()
        q_prop, p_prop = o[2], o[3]
        logw_prop = o[4]
        accep_rate_sum += o[5]

    return(q_prop,accep_rate_sum/L)