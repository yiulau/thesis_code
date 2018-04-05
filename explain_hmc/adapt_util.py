import math, numpy
from torch.autograd import Variable

def find_reasonable_ep(q,p,H_fun,integrator):
    # integrator can be leapfrog or gleapfrog using any possible metric
    ep = 1
    qprime,pprime = integrator(q,p,ep,H_fun)
    a = 2 * (-H_fun(qprime,pprime,True) + H_fun(q,p,True) > math.log(0.5)) - 1
    while a * (-H_fun(qprime, pprime,True) + H_fun(q, p,True)) > (-a * math.log(2)):
        ep = math.exp(a) * ep
        qprime,pprime = integrator(q,p,ep,H_fun)
    return(ep)


def dual_averaging_ep(sampler_onestep,generate_momentum,H_fun,integrator,q,
                      tune_l=2000, time=1.4, gamma=0.05, t_0=10, kappa=0.75, target_delta=0.65):
    # sampler_onestep should take an q pytorch variable and returns the next accepted q variable as well as acceptance_rate
    # find_reasonable_ep, should only depend on
    p = Variable(generate_momentum(q))
    ep = find_reasonable_ep(q,p,H_fun,integrator)
    mu = numpy.log(10 * ep)
    bar_ep_i = 1
    bar_H_i = 0

    store_ep = numpy.zeros(tune_l)
    for i in range(tune_l):
        num_step = max(1,round(time/ep))
        out = sampler_onestep(ep, num_step , q, integrator, H_fun)
        alpha = out[1]
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