import torch, numpy, math
from explicit.genleapfrog_ult_util import eigen

# move to abstract_class_V
# dphidq = dVdq + log(det(Sigma(q)))




def genleapfrog_wrap(alpha,delta,V,T):
    def inside(q,p,ep,pi):
        return generalized_leapfrog(q,p,ep,alpha,delta,V,T)
    return(inside)

def generalized_leapfrog(q,p,epsilon,alpha,delta,V,T):
    # input output point object
    #lam,Q = eigen(getH(q,V).data)
    #dH = getdH(q,V)
    #dV = getdV(q,V,True)
    # can take anything but should output tensor
    dV,H_,dH = V.getdH_tensor(q)
    lam, Q = eigen(H_)
    #p.data = p.data - epsilon * 0.5 * T.dphidq(lam,alpha,dH,Q,dV.data)
    # dphidq outputs and inputs takes flattened gradient in flattened form
    p.flattened_tensor -= epsilon * 0.5 * T.dphidq(lam,alpha,dH,Q,dV)
    p.loadfromflatten()
    #p.sum_load(T.dphidq(lam,alpha,dH,Q,dV.data).multiply_scalx(-epsilon*0.5))
    rho = p.flattened_tensor.clone()
    pprime = p.flattened_tensor.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        # dtaudq returns gradient in flattened form
        pprime = rho - epsilon * 0.5 * T.dtaudq(p.flattened_tensor,dH,Q,lam,alpha)
        deltap = torch.max(torch.abs(p.flattened_tensor-pprime))
        p.flattened_tensor.copy_(pprime)
        p.loadfromflatten()
        count = count + 1

    #sigma = Variable(q.data.clone(),requires_grad=True)
    sigma = q.point_clone()
    #qprime = q.data.clone()
    qprime = q.flattened_tensor.clone()
    deltaq = delta + 0.5

    _,H_ = V.getH_tensor(sigma)
    olam,oQ = eigen(H_)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = V.getH_tensor(q)
        lam,Q = eigen(H_)
        qprime = sigma.flattened_tensor + 0.5 * epsilon * T.dtaudp(p.flattened_tensor,alpha,olam,oQ) + \
                 0.5 * epsilon* T.dtaudp(p,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.flattened_tensor.copy(qprime)
        q.loadfromflatten()
        count = count + 1

    dV,H_,dH = V.getdH_tensor(q)
    lam,Q = eigen(H_)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    #p.data = p.data - 0.5 * T.dtaudq(p.data,dH,Q,lam,alpha) * epsilon
    p.flattened_tensor -= 0.5 * epsilon * T.dtaudq(p.flattened_tensor,dH,Q,lam,alpha)
    p.loadfromflatten()
    p.flattened_tensor -=0.5 * epsilon * T.dphidq(lam,alpha,dH,Q,dV)
    p.loadfromflatten()
    #p.data = p.data - 0.5 * T.dphidq(lam,alpha,dH,Q,dV.data) * epsilon
    return(q,p)

def rmhmc_step(initq,H,epsilon,L,alpha,delta,V,T):
    #q = Variable(initq.data.clone(), requires_grad=True)
    q = initq.point_clone()
    _, H_ = V.getH(q)
    lam, Q = eigen(H_)
    #lam,Q = eigen(getH(q,V).data)
    #p = Variable(generate_momentum(alpha,lam,Q))
    p = T.generate_momentum(q)
    current_H = H(q,p)

    for _ in range(L):
        out = generalized_leapfrog(q,p,epsilon,alpha,delta,V,T)
        q.flattened_tensor.copy_(out[0])
        p.flattened_tensor.copy_(out[1])
        q.loadfromflatten()
        p.loadfromflatten()
    proposed_H = H(q,p)
    u = numpy.random.rand(1)
    diff = current_H - proposed_H
    if (abs(diff) > 1000):
        divergent = True
    else:
        divergent = False
    accept_rate = math.exp(min(0,diff))
    if u < accept_rate:
        next_q = q
        accepted = True
    else:
        next_q = initq
        accepted = False
    return(next_q,divergent,accept_rate,accepted)