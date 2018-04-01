import torch,math,numpy
from torch.autograd import Variable,grad

def logsumexp(a,b):
    # stable way to calculate logsumexp
    # input torch tensor
    # output torch tensor = log(exp(a)+exp(b))
    s = torch.max(a,b)
    out = s + torch.log(torch.exp(a-s) + torch.exp(b-s))
    return(out)

def leapfrog_ult(q,p,epsilon,H_fun):
    H = H_fun(q,p)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    q.data += epsilon * p.data
    H = H_fun(q,p)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    return(q,p)

def HMC_alt_ult(epsilon, L, current_q, leapfrog, H_fun):

    p = Variable(torch.randn(len(current_q)), requires_grad=False)
    q = Variable(current_q.data.clone(),requires_grad=True)
    current_H = H_fun(q, p).data[0]
    for _ in range(L):
        o = leapfrog(q, p, epsilon, H_fun)
        q.data, p.data = o[0].data, o[1].data

    proposed_H = H_fun(q, p).data[0]
    diff = current_H - proposed_H
    if (abs(diff) > 1000):
        return_q = current_q.data
        return_H = current_H
        accept_rate = 0
        accepted = False
        divergent = True
    else:
        accept_rate = math.exp(min(0,diff))
        divergent = False
        if (numpy.random.random(1) < accept_rate):
            accepted = True
            return_q = q.data
            return_H = proposed_H
        else:
            accepted = False
            return_q = current_q.data
            return_H = current_H
    return(return_q,return_H,accepted,accept_rate,divergent)