import math,numpy
from explicit.general_util import logsumexp
from time_diagnostics import time_diagnositcs
# all functions modify (q,p)


def abstract_leapfrog_ult(q,p,epsilon,Ham):
    # Input:
    # q, p point objects
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)
    # evaluate gradient 2 times
    # evaluate H 0 times

    p.flattened_tensor -= Ham.V.dq(q.flattened_tensor) * 0.5 * epsilon
    q.flattened_tensor += Ham.T.dp(p.flattened_tensor) * epsilon
    p.flattened_tensor -= Ham.V.dq(q.flattened_tensor) * 0.5 * epsilon
    p.load_flatten()
    q.load_flatten()

    return(q,p)

def abstract_leapfrog_window(q_left,p_left,q_right,p_right,epsilon,Ham,logw_old,qprop_old,pprop_old):
    # Input: q,p current (q,p) state in trajecory
    # q,p point objects
    # qprop_old, pprop_old current proposed states in trajectory
    # logw_old = -H(qprop_old,pprop_old,return_float=True)
    # evaluate gradient 2 times
    # evaluate H 1 time


    v = numpy.random.choice([-1, 1])

    if v<0:
        q_left,p_left = abstract_leapfrog_ult(q_left,p_left,v*epsilon,Ham)

        logw_prop = -Ham.evaluate(q_left, p_left)
        accep_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
        u = numpy.random.rand(1)[0]
        if u < accep_rate:
            qprop = q_left
            pprop = p_left
        else:
            qprop = qprop_old
            pprop = pprop_old
            logw_prop = logw_old
    else:
        q_right,p_right = abstract_leapfrog_ult(q_right,p_right,v*epsilon,Ham)
        logw_prop = -Ham.evaluate(q_right, p_right)
        accep_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
        u = numpy.random.rand(1)[0]
        if u < accep_rate:
            qprop = q_right
            pprop = p_right
        else:
            qprop = qprop_old
            pprop = pprop_old
            logw_prop = logw_old


    #q,p = abstract_leapfrog_ult(q,p,v*epsilon,Ham)

    #print(logw_prop)
    #print(logw_old)
    #accep_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
    #u = numpy.random.rand(1)[0]
    #if u < accep_rate:
    #    qprop = q
    #    pprop = p
    #else:
    #    qprop = qprop_old
    #    pprop = pprop_old
    #    logw_prop = logw_old
    return(q_left,p_left,q_right,p_right,qprop,pprop,logw_prop,accep_rate)

def abstract_HMC_alt_ult(epsilon, L, current_q, leapfrog,Ham,evol_t=None,careful=True):
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
    # evaluate gradient L*2 times
    # evluate H 1 time
    if not L==None and not evol_t==None:
        raise ValueError("L contradicts with evol_t")
    Ham.diagnostics = time_diagnositcs()
    p = Ham.T.generate_momentum()
    q = current_q
    current_H = Ham.evaluate(q,p)
    for i in range(L):
        q,p = leapfrog(q, p, epsilon,Ham)
        if careful:
            temp_H = Ham.evaluate(q, p)
            if(abs(temp_H-current_H)>1000):
                return_q = current_q
                return_H = current_H
                accept_rate = 0
                accepted = False
                divergent = True
                return (return_q, return_H, accepted, accept_rate, divergent,i)



    proposed_H = Ham.evaluate(q,p)
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
    Ham.diagnostics.update_time()
    return(return_q,return_H,accepted,accept_rate,divergent)

def abstract_HMC_alt_windowed(epsilon, L, current_q, leapfrog_window, Ham,evol_t=None):
    # evaluate gradient 2*L times
    # evluate H function L times
    if not L==None and not evol_t==None:
        raise ValueError("L contradicts with evol_t")
    Ham.diagnostics = time_diagnositcs()
    p = Ham.T.generate_momentum()
    q = current_q
    logw_prop = -Ham.evaluate(q,p)
    q_prop = q.point_clone()
    p_prop = p.point_clone()
    q_left,p_left = q.point_clone(),p.point_clone()
    q_right,p_right = q.point_clone(), p.point_clone()

    accep_rate_sum = 0
    for _ in range(L):
        o = abstract_leapfrog_window(q_left, p_left,q_right,p_right,epsilon, Ham,logw_prop,q_prop,p_prop)
        q_left,p_left,q_right,p_right = o[0:4]
        q_prop, p_prop = o[4], o[5]
        logw_prop = o[6]
        #print(o[7])

        #accep_rate_sum += o[5]

    Ham.diagnostics.update_time()
    #return(q_prop,accep_rate_sum/L)
    return(q_prop,o[7])