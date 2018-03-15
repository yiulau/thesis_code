import torch
from torch.autograd import Variable
import numpy
import time, cProfile, math

dim = 2
chain_l = 1000
burn_in = 100
max_tdepth = 5


def pi(q, p):
    h = torch.dot(q, q) * 0.5 + torch.dot(p, p) * 0.5
    return(h)

def logsumexp(a, b):
    s = max(a,b)
    output = s + math.log((math.exp(a-s) + math.exp(b-s)))
    return(output)

def NUTS(q_init,epsilon,pi,leapfrog,NUTS_criterion):
    p = Variable(torch.randn(len(q_init)),requires_grad=False)
    q_left = Variable(q_init.data.clone(),requires_grad=True)
    q_right = Variable(q_init.data.clone(),requires_grad=True)
    p_left = Variable(p.data.clone(),requires_grad=False)
    p_right = Variable(p.data.clone(),requires_grad=False)
    j = 0
    q_prop = Variable(q_init.data.clone(),requires_grad=True)
    log_w = -pi(q_init.data,p.data)
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, s_prime, log_w_prime = BuildTree(q_left, p_left, -1, j, epsilon, leapfrog, pi,
                                                                            NUTS_criterion)
        else:
            _, _, q_right, p_right, q_prime, s_prime, log_w_prime = BuildTree(q_right, p_right, 1, j, epsilon, leapfrog, pi,
                                                                              NUTS_criterion)
        if s_prime:
            accep_rate = min(1,math.exp(log_w_prime-log_w))
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prop.data = q_prime.data.clone()
        log_w = logsumexp(log_w,log_w_prime)
        s = s_prime and NUTS_criterion(q_left,q_right,p_left,p_right)
        j = j + 1
        s = s and (j<max_tdepth)
    return(q_prop,j)

def BuildTree(q,p,v,j,epsilon,leapfrog,pi,NUTS_criterion):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,pi)
        log_w_prime = -pi(q_prime.data,p_prime.data)
        return q_prime, p_prime, q_prime, p_prime, q_prime, True, log_w_prime
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime = BuildTree(q, p, v, j - 1, epsilon, leapfrog, pi, NUTS_criterion)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,s_dprime,log_w_dprime = BuildTree(q_left,p_left,v,j-1,epsilon,leapfrog,pi,NUTS_criterion)
            else:
                _, _, q_right, p_right, q_dprime, s_dprime, log_w_dprime = BuildTree(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, pi, NUTS_criterion)
            accep_rate = min(1,math.exp(log_w_dprime-logsumexp(log_w_prime,log_w_dprime)))
            u = numpy.random.rand(1)[0]
            if u < accep_rate:
                q_prime.data = q_dprime.data.clone()
            s_prime = s_dprime and NUTS_criterion(q_left,q_right,p_left,p_right)
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime

def leapfrog(q,p,epsilon,pi):
    p_prime = Variable(p.data.clone(),requires_grad=False)
    q_prime = Variable(q.data.clone(),requires_grad=True)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5 * epsilon
    q_prime.grad.data.zero_()
    q_prime.data += epsilon * p_prime.data
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5 * epsilon
    q_prime.grad.data.zero_()
    return(q_prime, p_prime)



def NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue going
    # False = stops
    o = (torch.dot(q_right.data-q_left.data,p_right.data) >=0) and \
        (torch.dot(q_right.data-q_left.data,p_left.data) >=0)
    return(o)

q = Variable(torch.randn(dim),requires_grad=True)

v = -1

epsilon = 0.11

store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = NUTS(q,0.12,pi,leapfrog,NUTS_criterion)
    store[i,] = out[0].data # turn this on when using Nuts
    q.data = out[0].data # turn this on when using nuts

total = time.time() - begin
print("total time is {}".format(total))

store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
print(empCov)
print(emmean)
