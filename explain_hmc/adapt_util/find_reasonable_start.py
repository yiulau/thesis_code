import math
def find_reasonable_ep(tuning_obj):
    # integrator can be leapfrog or gleapfrog using any possible metric

    q,p = tuning_obj.cur_q,tuning_obj.cur_p
    Ham = tuning_obj.Ham
    integrator = tuning_obj.integrator
    ep = 1
    H_cur = Ham.evaluate(q,p)
    qprime,pprime = integrator(q,p,ep)

    a = 2 * (-Ham.evaluate(qprime,pprime) + H_cur > math.log(0.5)) - 1
    while a * (-Ham.evaluate(qprime,pprime) + H_cur) > (-a * math.log(2)):
        ep = math.exp(a) * ep
        qprime,pprime = integrator(q,p,ep)
    return(ep)





