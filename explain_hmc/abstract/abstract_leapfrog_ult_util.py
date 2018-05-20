import math,numpy
from general_util.time_diagnostics import time_diagnositcs
from abstract.abstract_genleapfrog_ult_util import gleapfrog_stat
# all functions modify (q,p)


def abstract_leapfrog_ult(q,p,epsilon,Ham):
    # Input:
    # q, p point objects
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)
    # evaluate gradient 2 times
    # evaluate H 0 times
    #print("yes")
    #print(Ham.evaluate(q,p))
    #out = {"q_tensor":q.flattened_tensor.clone(),"p_tensor":p.flattened_tensor.clone()}
    #import pickle
    #with open('debugqp.pkl', 'wb') as f:
    #    pickle.dump(out, f)
    #exit()
    p.flattened_tensor -= Ham.V.dq(q.flattened_tensor) * 0.5 * epsilon
    #print("first p abstract{}".format(p.flattened_tensor))
    #print("first H abstract {}".format(Ham.evaluate(q,p)))
    q.flattened_tensor += Ham.T.dp(p.flattened_tensor) * epsilon
    #print("first q abstract {}".format(q.flattened_tensor))
    #print("second H abstract {}".format(Ham.evaluate(q,p)))
    p.flattened_tensor -= Ham.V.dq(q.flattened_tensor) * 0.5 * epsilon
    #print("second p abstract {}".format(p.flattened_tensor))
    #print("final q abstract {}".format(q.flattened_tensor))
    p.load_flatten()
    q.load_flatten()
    #print(Ham.evaluate(q,p))
    #exit()
    return(q,p,gleapfrog_stat())



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

    else:
        q_right,p_right = abstract_leapfrog_ult(q_right,p_right,v*epsilon,Ham)
        logw_prop = -Ham.evaluate(q_right, p_right)

    if abs(logw_prop-logw_old)>1000:
        accept_rate = 0
        divergent = True

    else:
        # uniform progressive sampling
        #accept_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
        # baised progressive sampling
        accept_rate = math.exp(min(0,logw_prop-logw_old))
        divergent = False
    u = numpy.random.rand(1)[0]
    if u < accept_rate:
        qprop = q_right
        pprop = p_right
        accepted = True
    else:
        qprop = qprop_old
        pprop = pprop_old
        logw_prop = logw_old
        accepted = False



    return(q_left,p_left,q_right,p_right,qprop,pprop,logw_prop,divergent,accepted,accept_rate)

def abstract_HMC_alt_ult(epsilon, L, init_q,Ham,evol_t=None,careful=True):
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
    if not L is None and not evol_t is None:
        raise ValueError("L contradicts with evol_t")
    Ham.diagnostics = time_diagnositcs()
    if not evol_t is None:
        pass
    divergent = False
    num_transitions = L
    q = init_q.point_clone()
    init_p = Ham.T.generate_momentum(q)
    p = init_p.point_clone()
    current_H = Ham.evaluate(q,p)
    for i in range(L):
        q,p,_ = Ham.integrator(q, p, epsilon,Ham)
        if careful:
            temp_H = Ham.evaluate(q, p)
            if(abs(temp_H-current_H)>1000 ):
                return_q = init_q
                return_H = current_H
                accept_rate = 0
                accepted = False
                divergent = True
                return_p = None
                num_transitions = i

    if not divergent:
        proposed_H = Ham.evaluate(q,p)
        if (abs(current_H - proposed_H) > 1000):
            return_q = init_q
            return_p = None
            return_H = current_H
            accept_rate = 0
            accepted = False
            divergent = True

        else:
            accept_rate = math.exp(min(0,current_H - proposed_H))
            divergent = False
            if (numpy.random.random(1) < accept_rate):
                accepted = True
                return_q = q
                return_p = p
                return_H = proposed_H
            else:
                accepted = False
                return_q = init_q
                return_p = init_p
                return_H = current_H
    Ham.diagnostics.update_time()
    return(return_q,return_p,init_p,return_H,accepted,accept_rate,divergent,num_transitions)

def abstract_HMC_alt_windowed(epsilon, L, current_q, Ham,evol_t=None,careful=True):
    # evaluate gradient 2*L times
    # evluate H function L times
    if not L==None and not evol_t==None:
        raise ValueError("L contradicts with evol_t")
    Ham.diagnostics = time_diagnositcs()
    if not evol_t is None:
        pass
    divergent = False
    num_transitions = L
    accepted = False
    q = current_q
    p_init = Ham.T.generate_momentum(q)
    p = p_init.point_clone()
    logw_prop = -Ham.evaluate(q,p)
    current_H = -logw_prop
    q_prop = q.point_clone()
    p_prop = p.point_clone()
    q_left,p_left = q.point_clone(),p.point_clone()
    q_right,p_right = q.point_clone(), p.point_clone()

    for i in range(L):
        o = abstract_leapfrog_window(q_left, p_left,q_right,p_right,epsilon, Ham,logw_prop,q_prop,p_prop)
        q_left,p_left,q_right,p_right = o[0:4]
        q_prop, p_prop = o[4], o[5]
        logw_prop = o[6]
        divergent = o[7]
        accepted = o[8] or accepted
        accept_rate = o[9]
        if careful:
            if divergent:
                num_transitions = i
                break
        #print(o[7])

        #accep_rate_sum += o[5]


    #return(q_prop,accep_rate_sum/L)
    return(q_prop,p_prop,p_init,-logw_prop,accepted,accept_rate,divergent,L)





def windowerize(integrator):
    def windowed_integrator(q_left, p_left, q_right, p_right, epsilon, Ham, logw_old, qprop_old, pprop_old):
        # Input: q,p current (q,p) state in trajecory
        # q,p point objects
        # qprop_old, pprop_old current proposed states in trajectory
        # logw_old = -H(qprop_old,pprop_old,return_float=True)
        # evaluate gradient 2 times
        # evaluate H 1 time

        v = numpy.random.choice([-1, 1])

        if v < 0:
            q_left, p_left,stat = integrator(q_left, p_left, v * epsilon, Ham)
            divergent = stat.divergent
            logw_prop = -Ham.evaluate(q_left, p_left)

        else:
            q_right, p_right,stat = integrator(q_right, p_right, v * epsilon, Ham)
            divergent = stat.divergent
            logw_prop = -Ham.evaluate(q_right, p_right)

        if (abs(logw_prop - logw_old) > 1000 or divergent):
            accept_rate = 0
            divergent = True

        else:
            # uniform progressive sampling
            # accept_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
            # baised progressive sampling
            accept_rate = math.exp(min(0, logw_prop - logw_old))
            divergent = False
        u = numpy.random.rand(1)[0]
        if u < accept_rate:
            qprop = q_right
            pprop = p_right
            accepted = True
        else:
            qprop = qprop_old
            pprop = pprop_old
            logw_prop = logw_old
            accepted = False

        return (q_left, p_left, q_right, p_right, qprop, pprop, logw_prop, divergent, accepted, accept_rate)
    return(windowed_integrator)