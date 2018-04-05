import math, numpy, torch
from torch.autograd import Variable

def find_reasonable_ep(q,p,H_fun,integrator):
    # integrator can be leapfrog or gleapfrog using any possible metric
    ep = 1
    H_cur = H_fun(q,p,True)
    qprime,pprime = integrator(Variable(q.data.clone(),requires_grad=True),Variable(p.data.clone()),ep,H_fun)
    a = 2 * (-H_fun(qprime,pprime,True) + H_cur > math.log(0.5)) - 1
    while a * (-H_fun(qprime, pprime,True) + H_cur) > (-a * math.log(2)):
        ep = math.exp(a) * ep
        qprime,pprime = integrator(Variable(q.data.clone(),requires_grad=True),Variable(p.data.clone()),ep,H_fun)
    return(ep)


def dual_averaging_ep(sampler_onestep,generate_momentum,H_fun,integrator,q,
                      tune_l=2000, time=1.4, gamma=0.05, t_0=10, kappa=0.75, target_delta=0.65):
    # sampler_onestep should take an q pytorch variable and returns the next accepted q variable as well as acceptance_rate
    # find_reasonable_ep, should only depend on
    # store_ep numpy array storing the epsilons
    p = Variable(generate_momentum(q))
    ep = find_reasonable_ep(q,p,H_fun,integrator)
    mu = math.log(10 * ep)
    bar_ep_i = 1
    bar_H_i = 0

    store_ep = numpy.zeros(tune_l)
    for i in range(tune_l):
        num_step = max(1,round(time/ep))
        out = sampler_onestep(ep, num_step , q, integrator, H_fun,generate_momentum)
        alpha = out[3]
        bar_ep_i, bar_H_i = adapt_ep(alpha,bar_H_i,t_0,i,target_delta,gamma,bar_ep_i,kappa,mu)
        store_ep[i] = bar_ep_i
        ep = bar_ep_i
        q.data = out[0].data
    return(store_ep,q)

def adapt_ep(alpha,bar_H_i,t_0,i,target_delta,gamma,bar_ep_i,kappa,mu):
    bar_H_i = (1 - 1 / (i + 1 + t_0)) * bar_H_i + (1 / (i + 1 + t_0)) * (target_delta - alpha)
    logep = mu - math.sqrt(i + 1) / gamma * bar_H_i
    logbarep = math.pow(i + 1, -kappa) * logep + (1 - math.pow(i + 1, -kappa)) * math.log(bar_ep_i)
    bar_ep_i = math.exp(logbarep)

    return(bar_ep_i,bar_H_i)

def welford(next_sample,sample_counter,m_,m_2,diag):
    # next_sample pytorch tensor
    # diag boolean variable if true m_2 is the accumulative varinces
    #                       if false m_2 is the accumulative covars
    sample_counter = sample_counter + 1
    delta = (next_sample-m_)
    m_ += delta/sample_counter
    # torch.ger(x,y) = x * y^T
    if diag:
        m_2 += (next_sample-m_) * delta
    else:
        m_2 += torch.ger((next_sample-m_),delta)
    return(m_,m_2,sample_counter)


def full_adapt(sampler_onestep,generate_momentum,H_fun,integrator,q,
                      tune_l=2000, time=1.4, gamma=0.05, t_0=10, kappa=0.75, target_delta=0.65,covar=None):
    # sampler_onestep should take an q pytorch variable and returns the next accepted q variable as well as acceptance_rate
    # find_reasonable_ep, should only depend on
    # store_ep numpy array storing the epsilons
    # covar can be None,dense,or diag
    # adapt both epsilon and covariances
    p = Variable(generate_momentum(q))
    ep = find_reasonable_ep(q,p,H_fun,integrator)
    mu = math.log(10 * ep)
    bar_ep_i = 1
    bar_H_i = 0
    store_ep = numpy.zeros(tune_l)
    if covar !=None:
        for i in range(tune_l):
            num_step = max(1,round(time/ep))
            out = sampler_onestep(ep, num_step , q, integrator, H_fun,generate_momentum)
            alpha = out[3]
            bar_ep_i, bar_H_i = adapt_ep(alpha,bar_H_i,t_0,i,target_delta,gamma,bar_ep_i,kappa,mu)
            store_ep[i] = bar_ep_i
            ep = bar_ep_i
            q.data = out[0].data
        return(store_ep,q)
    else:
        window_width = 25
        ini_buffer = 75
        end_buffer = 50
        counter_ep = 0
        counter_cov = 0
        dim = len(q)
        m_ = torch.zeros(dim)
        if covar == "dense":
            m_2 = torch.zeros((dim,dim))
        else:
            m_2 = torch.zeros(dim)
        for i in range(tune_l):
            # updates epsilon only in the beginning and at the end
            if i < ini_buffer or i >= tune_l - end_buffer:
                num_step = max(1, round(time / ep))
                out = sampler_onestep(ep, num_step, q, integrator, H_fun, generate_momentum)
                alpha = out[3]
                bar_ep_i, bar_H_i = adapt_ep(alpha, bar_H_i, t_0, counter_ep, target_delta, gamma, bar_ep_i, kappa, mu)
                store_ep[i] = bar_ep_i
                ep = bar_ep_i
                counter_ep += 1
                q.data = out[0].data
            else:
                if i in update_metric_and_eplist:
                    num_step = max(1, round(time / ep))
                    out = sampler_onestep(ep, num_step, q, integrator, H_fun, generate_momentum)
                    alpha = out[3]
                    bar_ep_i, bar_H_i = adapt_ep(alpha, bar_H_i, t_0, counter_ep, target_delta, gamma, bar_ep_i, kappa,
                                                 mu)
                    store_ep[i] = bar_ep_i
                    ep = bar_ep_i
                    counter_ep += 1
                    q.data = out[0].data
                    m_, m_2, counter_cov = welford(q.data, counter_cov, m_, m_2, True)
                    generate_momentum,H_fun=update_metric(generate_momentum,H_fun)
                else:
                    m_, m_2, counter_cov = welford(q.data, counter_cov, m_, m_2, True)

    return(store_ep,q)