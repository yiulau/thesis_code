import torch,math,numpy
from torch.autograd import Variable
from explicit.general_util import logsumexp

def leapfrog_ult(q,p,epsilon,H_fun):
    # Input:
    # q, p pytorch variables
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)

    H = H_fun(q,p,return_float=False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    #print("first p exact{}".format(p.data))
    #print("first H exact{}".format(H_fun(q,p,True)))
    q.grad.data.zero_()
    q.data += epsilon * p.data
    #print("first q exact {}".format(q.data))
    #print("second H exact {}".format(H_fun(q,p,True)))
    H = H_fun(q,p,return_float=False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    #print("second p exact {}".format(p.data))
    #print("final q exact {}".format(q.data))
    q.grad.data.zero_()

    return(q,p)

def leapfrog_ult_tensor(q,p,epsilon,V,T):
    # (q,p) are pytorch tensors
    # q,p flat tensors
    # V.dq , T.dp takes (flat) tensors returns tensor
    p -= V.dq(q) * 0.5 * epsilon
    q += T.dp(p) * epsilon
    p -= V.dq(q) * 0.5 * epsilon
    return(q,p)


def leapfrog_window(q,p,epsilon,H_fun,logw_old,qprop_old,pprop_old):
    # Input: q,p current (q,p) state in trajecory
    # qprop_old, pprop_old current proposed states in trajectory
    # logw_old = -H(qprop_old,pprop_old,return_float=True)
    H = H_fun(q,p,False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    q.data += epsilon * p.data
    H = H_fun(q,p,False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    logw_prop = -H_fun(q,p,True)
    accep_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
    u = numpy.random.rand(1)[0]
    if u < accep_rate:
        qprop = q
        pprop = p
    else:
        qprop = qprop_old
        pprop = pprop_old
        logw_prop = logw_old
    return(q,p,qprop,pprop,logw_prop,accep_rate)

def leapfrog_window_tensor(q,p,epsilon,V,T,H,logw_old,qprop_old,pprop_old):
    # Input: q,p current (q,p) state in trajecory
    # qprop_old, pprop_old current proposed states in trajectory
    # logw_old = -H(qprop_old,pprop_old,return_float=True)

    p -= V.dq(q) * 0.5 * epsilon
    q += epsilon * T.dp(p)
    p -= V.dq(q) * 0.5 * epsilon
    logw_prop = -H(q,p)
    accep_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
    u = numpy.random.rand(1)[0]
    if u < accep_rate:
        qprop = q
        pprop = p
    else:
        qprop = qprop_old
        pprop = pprop_old
        logw_prop = logw_old
    return(q,p,qprop,pprop,logw_prop,accep_rate)

def HMC_alt_ult(epsilon, L, current_q, leapfrog, H_fun,generate_momentum):
    # Input:
    # current_q Pytorch Variable
    # H_fun(q,p,return_float) returns Pytorch Variable or float
    # generate_momentum(q) returns pytorch tensor
    # Output:
    # accept_rate: float - probability of acceptance
    # accepted: Boolean - True if proposal is accepted, False otherwise
    # divergent: Boolean - True if the end of the trajectory results in a divergent transition
    # return_q  pytorch Variable (! not tensor)
    q = Variable(current_q.data.clone(),requires_grad=True)
    p = Variable(generate_momentum(q), requires_grad=False)
    current_H = H_fun(q, p,True)

    for _ in range(L):
        o = leapfrog(q, p, epsilon, H_fun)
        q.data, p.data = o[0].data, o[1].data

    proposed_H = H_fun(q, p,True)
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
    print("current_H {},propsed_H{}".format(current_H,proposed_H))
    print("accept_rate {}".format(accept_rate))
    exit()
    return(return_q,return_H,accepted,accept_rate,divergent)

def HMC_alt_ult_tensor(epsilon, L, current_q, leapfrog, V,T,H):
    # Input:
    # current_q Pytorch tensor
    # H_fun(q,p,return_float) returns Pytorch Variable or float
    # Output:
    # accept_rate: float - probability of acceptance
    # accepted: Boolean - True if proposal is accepted, False otherwise
    # divergent: Boolean - True if the end of the trajectory results in a divergent transition
    # return_q  pytorch tensor
    q = current_q.clone()
    p = T.generate_momentum(q)
    current_H = H(q, p)

    for _ in range(L):
        o = leapfrog(q, p, epsilon, V,T)
        q, p = o[0], o[1]

    proposed_H = H(q, p)
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

    return(return_q,return_H,accepted,accept_rate,divergent)
def HMC_alt_windowed(epsilon, L, current_q, leapfrog_window, H_fun):
    p = Variable(torch.randn(len(current_q)), requires_grad=False)
    q = Variable(current_q.data, requires_grad=True)
    logw_prop = -H_fun(q, p,True)
    q_prop = q.clone()
    p_prop = p.clone()
    accep_rate_sum = 0
    num_divergent = 0
    for _ in range(L):
        o = leapfrog_window(q, p, epsilon, H_fun,logw_prop,q_prop,p_prop)
        q.data, p.data = o[0].data.clone(), o[1].data.clone()
        q_prop, p_prop = o[2], o[3]
        diff = abs(logw_prop-o[4])
        if abs(diff)>1000:
            num_divergent +=1
        logw_prop = o[4]

        accep_rate_sum += o[5]

    return(q_prop,accep_rate_sum/L,num_divergent)


def HMC_alt_windowed_tensor(epsilon, L, current_q, leapfrog_window, V,T,H):
    # input and output are both tensors
    q = current_q.clone()
    p = T.generate_momentum(q)
    logw_prop = -H(q, p)
    q_prop = q.clone()
    p_prop = p.clone()
    accep_rate_sum = 0
    num_divergent = 0
    for _ in range(L):
        o = leapfrog_window(q, p, epsilon, V,T,H, logw_prop, q_prop, p_prop)
        q, p = o[0], o[1]
        q_prop, p_prop = o[2], o[3]
        diff = abs(logw_prop - o[4])
        if abs(diff) > 1000:
            num_divergent += 1
        logw_prop = o[4]

        accep_rate_sum += o[5]

    return (q_prop, accep_rate_sum / L, num_divergent)