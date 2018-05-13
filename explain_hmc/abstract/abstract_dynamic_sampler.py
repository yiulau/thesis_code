import torch, numpy, math
from explicit.general_util import logsumexp, stable_sum
from general_util.time_diagnostics import time_diagnositcs

def abstract_NUTS(q_init,epsilon,Ham,max_tdepth=5):
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
    accepted = False
    divergent = False
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, p_prime,s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q_left, p_left, -1, j, epsilon, Ham,H_0
                                                                            )
        else:
            _, _, q_right, p_right, q_prime, p_prime, s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q_right, p_right, 1, j, epsilon, Ham,H_0
                                                                              )
        if s_prime:
            accept_rate = math.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                accepted = accepted or True
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()

        log_w = logsumexp(log_w,log_w_prime)
        s = s_prime and abstract_NUTS_criterion(q_left,q_right,p_left,p_right)
        j = j + 1
        s = s and (j<max_tdepth)
        num_div += num_div_prime
        Ham.diagnostics.update_time()
    if num_div >0:
        divergent = True
        p_prop = None

    return(q_prop,p_prop,p_init,-log_w,accepted,accept_rate,divergent,j)