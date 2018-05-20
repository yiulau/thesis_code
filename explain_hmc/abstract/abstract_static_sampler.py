import math,numpy
from general_util.time_diagnostics import time_diagnositcs
from abstract.abstract_genleapfrog_ult_util import gleapfrog_stat


def abstract_static_one_step(epsilon, init_q,Ham,evolve_L=None,evolve_t=None,log_obj=None,alpha=None):
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


    if not evolve_L is None and not evolve_t is None:
        raise ValueError("L contradicts with evol_t")
    assert evolve_L is None or evolve_t is None
    assert not (evolve_L is None and evolve_t is None)
    if not evolve_t is None:
        assert evolve_L is None
        evolve_L = round(evolve_t/epsilon)
    careful = True
    Ham.diagnostics = time_diagnositcs()
    divergent = False
    num_transitions = evolve_L
    q = init_q.point_clone()
    init_p = Ham.T.generate_momentum(q)
    p = init_p.point_clone()
    #print(q.flattened_tensor)
    #print(p.flattened_tensor)
    current_H = Ham.evaluate(q,p)

    print("startH {}".format(current_H))


    #newq,newp,stat = Ham.integrator(q, p, epsilon, Ham)
    #print(q.flattened_tensor)
    #print(p.flattened_tensor)
    #newH = Ham.evaluate(newq,newp)
    #print(newH)
    #exit()
    #print(type(evolve_L))
    #exit()
    #print(q.flattened_tensor)
    #print(p.flattened_tensor)
    #print("epsilon is {}".format(epsilon))
    for i in range(evolve_L):
        q, p, stat = Ham.integrator(q, p, epsilon, Ham)
        divergent = stat.divergent
        #print(q.flattened_tensor)
        #print(p.flattened_tensor)
        if careful:
            temp_H = Ham.evaluate(q, p)
            #print("H is {}".format(temp_H))
            if(abs(temp_H-current_H)>1000 or divergent):
                #print("yeye")
                #print(i)
                #print(temp_H)
                #print(current_H)
                #exit()
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
        #print(log_obj is None)
    endH = Ham.evaluate(q,p)
    accept_rate = math.exp(min(0, current_H - endH))
    #print("accept_rate {}".format(accept_rate))
    #print("endH {}".format(Ham.evaluate(q,p)))
    #exit()
    if not log_obj is None:
        log_obj.store.update({"prop_H":return_H})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"num_transitions":num_transitions})

    return(return_q,return_p,init_p,return_H,accepted,accept_rate,divergent,num_transitions)


def abstract_static_windowed_one_step(epsilon, init_q, Ham,evolve_L=None,evolve_t=None,careful=True,log_obj=None,alpha=None):
    # evaluate gradient 2*L times
    # evluate H function L times

    assert evolve_L is None or evolve_t is None
    if not evolve_L==None and not evolve_t==None:
        raise ValueError("L contradicts with evol_t")

    if not evolve_t is None:
        assert evolve_L is None
        evolve_L = round(evolve_t/epsilon)
    Ham.diagnostics = time_diagnositcs()
    divergent = False
    num_transitions = evolve_L
    accepted = False
    q = init_q
    p_init = Ham.T.generate_momentum(q)
    p = p_init.point_clone()
    logw_prop = -Ham.evaluate(q,p)
    current_H = -logw_prop
    q_prop = q.point_clone()
    p_prop = p.point_clone()
    q_left,p_left = q.point_clone(),p.point_clone()
    q_right,p_right = q.point_clone(), p.point_clone()

    for i in range(evolve_L):
        o = Ham.windowed_integrator(q_left, p_left,q_right,p_right,epsilon, Ham,logw_prop,q_prop,p_prop)
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
        if not divergent:
            num_transitions = evolve_L
        #print(o[7])

        #accep_rate_sum += o[5]


    #return(q_prop,accep_rate_sum/L)
    if not log_obj is None:
        log_obj.store.update({"prop_H":-logw_prop})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"num_transitons":num_transitions})

    return(q_prop,p_prop,p_init,-logw_prop,accepted,accept_rate,divergent,num_transitions)