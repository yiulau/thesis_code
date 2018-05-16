import math,numpy
def find_reasonable_ep(Ham):
    # integrator can be leapfrog or gleapfrog using any possible metric
    counter = 0
    #q,p = one_chain_obj.cur_q,one_chain_obj.cur_p
    q = Ham.V.q_point.point_clone()
    p = Ham.T.generate_momentum(q)

    integrator = Ham.integrator
    epsilon = 1
    H_cur = Ham.evaluate(q,p)
    qprime,pprime,_ = integrator(q=q.point_clone(),p=p.point_clone(),epsilon=epsilon,Ham=Ham)

    a = 2 * (-Ham.evaluate(qprime,pprime) + H_cur > math.log(0.5)) - 1
    while a * (-Ham.evaluate(qprime,pprime) + H_cur) > (-a * math.log(2)):
        epsilon = math.exp(a) * epsilon
        qprime,pprime,_ = integrator(q=q.point_clone(),p=p.point_clone(),epsilon=epsilon,Ham=Ham)

        counter +=1
        if counter > 100:
            raise ValueError("find_reasonable_ep takes too long. check")

    return(epsilon)


def find_min_epsilon(Ham):
    # don't need to change integrator. two cases:
    # for the case of leapfrog, both diag_cov and dense_cov are set to unitity covariance == unit_e
    reasonable_start = find_reasonable_ep(Ham)
    ep_list = [reasonable_start]
    keep_going = True
    counter = 0
    q = Ham.V.q_point.point_clone()
    p = Ham.T.generate_momentum(q)

    integrator = Ham.integrator
    H_cur = Ham.evaluate(q, p)

    #num_adapt_trajectory = 200
    #num_super_transitions = 500

    num_adapt_trajectory = 200
    hard_lower_limit_ep = 0.001
    while keep_going:
        accept_rate_list = []
        evolve_L = round(num_adapt_trajectory*numpy.random.uniform(0.1,0.9))
        for i in range(evolve_L):
            qprime, pprime, _=integrator(q=q.point_clone(), p=p.point_clone(), epsilon=ep_list[counter], Ham=Ham)
            H_prop = Ham.evaluate(qprime,pprime)
            accept_rate = math.exp(min(0,H_prop-H_cur))
            accept_rate_list.append(accept_rate)
            q = qprime
            p = pprime
            H_cur = H_prop
        ave_accept = numpy.mean(accept_rate_list)
        if ave_accept > 0.99:
            if ep_list[counter]>=ep_list[0]:
                ep_list.append(ep_list[counter]*2)
            else:
                keep_going = False
                chosen_ep = ep_list[counter]
        else:
            # impossible to stop at counter = 1 if counter=0 does not succeed
            if ep_list[counter]>ep_list[0]:
                keep_going = False
                chosen_ep = ep_list[counter-1]
            else:
                if ep_list[counter]/2 > hard_lower_limit_ep:
                    ep_list.append(ep_list[counter]/2)
                else:
                    raise ValueError("ep goes below hard limit, still not achieving desired average acceptance rate")

        counter += 1
    lower_ep = chosen_ep
    max_ep = max(ep_list)
    out = {"lower_ep":lower_ep,"max_ep":max_ep}
    return(out)

def find_max_epsilon(Ham,start_ep):
    # find_max_ep
    # has to be > min_epsilon
    # given information from find_min_epsilon, (largest epsilon tried)
    # dobule until average acceptance rate is < 50 %
    # start_ep should be find_min_epsilon(Ham)["max_ep"]
    max_ep_reasonable_start = start_ep
    ep_list = [max_ep_reasonable_start]
    keep_going = True
    counter = 0
    q = Ham.V.q_point.point_clone()
    p = Ham.T.generate_momentum(q)

    integrator = Ham.integrator
    H_cur = Ham.evaluate(q, p)
    num_adapt_trajectory = 200
    keep_going = True

    while keep_going:
        accept_rate_list = []

        evolve_L = round(num_adapt_trajectory*numpy.random.uniform(0.1,0.9))
        for i in range(evolve_L):
            qprime, pprime, _ = integrator(q=q.point_clone(), p=p.point_clone(), epsilon=ep_list[counter], Ham=Ham)
            H_prop = Ham.evaluate(qprime, pprime)
            accept_rate = math.exp(min(0, H_prop - H_cur))
            accept_rate_list.append(accept_rate)
            q = qprime
            p = pprime
            H_cur = H_prop
        ave_accept = numpy.mean(accept_rate_list)

        if ave_accept < 0.5:
            keep_going = False
            chosen_ep = ep_list[counter]
        else:
            ep_list.append(ep_list[counter]* 2)
        counter += 1
    upper_ep = chosen_ep
    return(upper_ep)


def find_min_max_epsilon(Ham):
    # don't need to change integrator. two cases:
    # for the case of leapfrog, both diag_cov and dense_cov are set to unitity covariance == unit_e
    reasonable_start = find_reasonable_ep(Ham)
    ep_list = [reasonable_start]
    keep_going = True
    counter = 0
    q = Ham.V.q_point.point_clone()
    p = Ham.T.generate_momentum(q)

    integrator = Ham.integrator
    H_cur = Ham.evaluate(q, p)
    num_adapt_trajectory = 200
    hard_lower_limit_ep = 0.001
    while keep_going:
        accept_rate_list = []
        evolve_L = round(num_adapt_trajectory*numpy.random.uniform(0.1,0.9))
        for i in range(evolve_L):
            qprime, pprime, _ = integrator(q=q.point_clone(), p=p.point_clone(), epsilon=ep_list[counter], Ham=Ham)
            H_prop = Ham.evaluate(qprime, pprime)
            accept_rate = math.exp(min(0, H_prop - H_cur))
            accept_rate_list.append(accept_rate)
            q = qprime
            p = pprime
            H_cur = H_prop
        ave_accept = numpy.mean(accept_rate_list)
        if ave_accept > 0.99:
            if ep_list[counter] >= ep_list[0]:
                ep_list.append(ep_list[counter] * 2)
            else:
                keep_going = False
                chosen_ep = ep_list[counter]
        else:
            # impossible to stop at counter = 1 if counter=0 does not succeed
            if ep_list[counter] > ep_list[0]:
                keep_going = False
                chosen_ep = ep_list[counter - 1]
            else:
                if ep_list[counter] / 2 > hard_lower_limit_ep:
                    ep_list.append(ep_list[counter] / 2)
                else:
                    raise ValueError("ep goes below hard limit, still not achieving desired average acceptance rate")

        counter += 1
    lower_ep = chosen_ep
    max_ep = max(ep_list)

    ep_list = [max_ep]
    keep_going = True
    while keep_going:
        accept_rate_list = []
        evolve_L = round(num_adapt_trajectory*numpy.random.uniform(0.1,0.9))
        for i in range(evolve_L):
            qprime, pprime, _ = integrator(q=q.point_clone(), p=p.point_clone(), epsilon=ep_list[counter], Ham=Ham)
            H_prop = Ham.evaluate(qprime, pprime)
            accept_rate = math.exp(min(0, H_prop - H_cur))
            accept_rate_list.append(accept_rate)
            q = qprime
            p = pprime
            H_cur = H_prop
        ave_accept = numpy.mean(accept_rate_list)

        if ave_accept < 0.5:
            keep_going = False
            chosen_ep = ep_list[counter]
        else:
            ep_list.append(ep_list[counter] * 2)
        counter += 1
    upper_ep = chosen_ep

    assert lower_ep < upper_ep
    out = {"lower_ep":lower_ep,"upper_ep":upper_ep}
    return(out)



#def find_min_max_delta(Ham):
