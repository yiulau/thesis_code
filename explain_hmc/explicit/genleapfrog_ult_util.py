import torch, numpy, math
from torch.autograd import Variable,grad
from explicit.general_util import logsumexp

def coth(x):
    # input and output is float
    return(1/numpy.asscalar(numpy.tanh(x)))
def coth_torch(x):
    # input and output are pytorch tensors
    return(1/torch.tanh(x))

def softabs_map(lam,alpha):
    # takes vector as input
    # returns a vector
    lower_softabs_thresh = 0.001
    upper_softabs_thresh = 18
    out = torch.zeros(len(lam))
    for i in range(len(lam)):
        alp_lam = lam[i] * alpha
        if (abs(alp_lam)<lower_softabs_thresh):
            out[i] = (1. + (1./3.) * alp_lam * alp_lam)/alpha
        elif (abs(alp_lam)>upper_softabs_thresh):
            out[i] = abs(lam[i])
        else:
            out[i] = lam[i] * coth(alp_lam)
    return(out)
def generate_momentum(alpha,lam,Q):
    # generate by multiplying st normal by QV^(0.5) where Sig = QVQ^T
    # Input: Q, lam - pytorch tensor
    # Output: out  - pytorch tensor , out ~ N(0,Sig)
    temp = torch.mm(Q,torch.diag(torch.sqrt(softabs_map(lam,alpha))))
    out = torch.mv(temp,torch.randn(len(lam)))
    return(out)

def eigen(H):
    # input must be of type tensor ** not variable
    # it should also be symmetric
    # returns lam, Q such that H = Q * diag(lam) Q^T = H
    out = torch.symeig(H,True)
    return(out[0],out[1])

def getdV(q,V,want_graph):
    # returns
    # return tensor
    g = grad(V(q), q, create_graph=want_graph)[0]
    return(g)
def getdV_tensor(q,V,want_graph):
    # takes tensor
    # return tensor
    qvar = Variable(q,requires_grad=True)
    g = grad(V(q), qvar, create_graph=want_graph)[0]
    return(g)
def getH(q,V):
    # output: H - Pytorch Variable
    # return flaat tensor
    # return g,H
    g = getdV(q,V,True)
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        H = Variable(torch.zeros(dim, dim)).cuda()
    else:
        H = Variable(torch.zeros(dim, dim))
    for i in range(dim):
        H[i, :] = grad(g[i], q, create_graph=True)[0]
    return(g,H)
def getH_tensor(q,V):
    # output: H - Pytorch tensor
    # return tensor tensor
    # return g,H
    qvar = Variable(q, requires_grad=True)
    g = grad(V(q), qvar, create_graph=True)[0]
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        H = (torch.zeros(dim, dim)).cuda()
    else:
        H = (torch.zeros(dim, dim))
    for i in range(dim):
        H[i, :] = grad(g[i], q, create_graph=True)[0].data
    g = g.data
    return(g,H)
########################
# modify getH,getdH so they return lower derivatives at the same time
# current code computes the derivatives twice, three times
# return tensor
# return g,H,dH
def getdH(q,V):
    # output: dH - pytorch tensor where
    g,H_ = getH(q,V)
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        dH = torch.zeros(dim,dim,dim).cuda()
    else:
        dH = torch.zeros(dim,dim,dim)
    for i in range(dim):
        for j in range(dim):
            dH[:, i, j] = grad(H_[i, j], q, create_graph=False,retain_graph=True)[0].data

    return(g,H_,dH)

def getdH_tensor(q,V):
    # input pytorch tensor
    # output: dH - pytorch tensor where

    qvar = Variable(q, requires_grad=True)
    g = grad(V(q), qvar, create_graph=True)[0]
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        H_ = Variable(torch.zeros(dim, dim)).cuda()
    else:
        H_ = Variable(torch.zeros(dim, dim))
    for i in range(dim):
        H_[i, :] = grad(g[i], q, create_graph=True)[0]
    g = g.data
    if q.data.type() == "torch.cuda.FloatTensor":
        dH = torch.zeros(dim,dim,dim).cuda()
    else:
        dH = torch.zeros(dim,dim,dim)
    for i in range(dim):
        for j in range(dim):
            dH[:, i, j] = grad(H_[i, j], q, create_graph=False,retain_graph=True)[0].data
    g = g.data
    H_ = H_.data
    return(g,H_,dH)
def getdH_specific(q,V):
    # inputs tensor
    # output: dH - pytorch tensor where
    # returns function dH_i(index), which computes dH_index
    # and H_
    qvar = Variable(q, requires_grad=True)
    g = grad(V(q), qvar, create_graph=True)[0]
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        H_ = Variable(torch.zeros(dim, dim)).cuda()
    else:
        H_ = Variable(torch.zeros(dim, dim))
    for i in range(dim):
        H_[i, :] = grad(g[i], q, create_graph=True)[0]
    def dH_i(i):
        dim = len(q)
        if q.data.type() == "torch.cuda.FloatTensor":
            dH = torch.zeros(dim,dim).cuda()
        else:
            dH = torch.zeros(dim,dim)
        for i in range(dim):
            for j in range(dim):
                dH[i, j] = grad(H_[i, j], q[i], create_graph=False,retain_graph=True)[0].data
        return(dH_i)
    return(g.data,dH_i,H_.data)
def dphidq(lam,alpha,dH,Q,dV):
    # returns pytorch tensor
    N = len(lam)
    Jm = J(lam,alpha,len(lam))
    R = torch.diag(1/(lam*coth_torch(alpha*lam)))
    M = torch.mm(Q,torch.mm(R*Jm,torch.t(Q)))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(torch.mm(M,dH[i,:,:])) + dV[i]
    return(delta)
def dphidq_specific(lam,alpha,dH_i,Q,dV):
    # returns pytorch tensor
    # dH_i[index] computes dH_index
    N = len(lam)
    Jm = J(lam,alpha,len(lam))
    R = torch.diag(1/(lam*coth_torch(alpha*lam)))
    M = torch.mm(Q,torch.mm(R*Jm,torch.t(Q)))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(torch.mm(M,dH_i[i])) + dV[i]
    return(delta)
def J(lam,alpha,length):
    jacobian_thresh = 0.001
    lower_softabs_thresh = 0.0001
    upper_softabs_thresh = 1000
    J = torch.zeros(length,length)
    i = 0
    while i < length:
        j =0
        while j <=i:
            dif = abs(lam[i]-lam[j])
            if dif < jacobian_thresh:

                # either i = j or lam[i] approx lam[j]
                alp_lam = lam[i] * alpha

                if abs(alp_lam) < lower_softabs_thresh:
                    # lam[i] too small

                    J[i,j] = (2./3.) * alp_lam * (1.- (2./15.)*alp_lam*alp_lam)
                elif alp_lam > upper_softabs_thresh:

                    # lam blows up
                    # 1 if lam > 0 , -1 otherwise
                    J[i,j] = 2*float(lam[i]>0)-1
                else:

                    J[i,j] = (coth(alpha * lam[i]) + lam[i] * (1 - (coth(alpha * lam[i]))**2) * alpha)
            else:

                J[i,j] = (lam[i]*coth(alpha*lam[i]) - lam[j]*coth(alpha*lam[j]))/(lam[i]-lam[j])
            J[j,i] = J[i,j]
            j = j + 1

        i = i + 1


    return(J)

def D(p,Q,lam,alpha):
    # output : Diag( (Q^T * p) / lam * coth(alpha * lam ))
    # Diagonal matrix such that ith entry is (Q^T * p)_i / ( lam_i * coth(alpha * lam_i))

    return(torch.diag(torch.mv(torch.t(Q),p)/softabs_map(lam,alpha)))

def dtaudq(p,dH,Q,lam,alpha):
    N = len(p)
    Jm = J(lam,alpha,len(p))
    #print("J {}".format(Jm))
    Dm = D(p,Q,lam,alpha)
    #print("J {}".format(Dm))
    M = torch.mm(Q,torch.mm(Dm,torch.mm(Jm,torch.mm(Dm,torch.t(Q)))))
    #print("M {}".format(M))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(-torch.mm(M,dH[i,:,:]))
    return (delta)
def dtaudq_specific(p,dH_i,Q,lam,alpha):
    N = len(p)
    Jm = J(lam,alpha,len(p))
    Dm = D(p,Q,lam,alpha)
    M = torch.mm(Q,torch.mm(Dm,torch.mm(Jm,torch.mm(Dm,torch.t(Q)))))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(-torch.mm(M,dH_i[i]))
    return (delta)
def dtaudp(p,alpha,lam,Q):
    return(Q.mv(torch.diag(1/softabs_map(lam,alpha)).mv((torch.t(Q).mv(p)))))

def genleapfrog_wrap(alpha,delta,V):
    def inside(q,p,ep,pi):
        return generalized_leapfrog(q,p,ep,alpha,delta,V)
    return(inside)

def generalized_leapfrog(q,p,epsilon,alpha,delta,V):
    #
    #lam,Q = eigen(getH(q,V).data)
    #dV,H_ = getH(q,V)
    #dH_i = getdH_specific(q,V,H_)
    #dV = getdV(q,V,True)
    dV,H_,dH = getdH(q,V)
    lam, Q = eigen(H_.data)
    p.data -= epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV.data)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime.copy_(rho - epsilon * 0.5 * dtaudq(p.data,dH,Q,lam,alpha))
        deltap = torch.max(torch.abs(p.data-pprime))
        p.data.copy_(pprime)
        count = count + 1

    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5

    _,H_ = getH(sigma,V)
    olam,oQ = eigen(H_.data)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = getH(q,V)
        lam,Q = eigen(H_.data)
        qprime.copy_(sigma.data + 0.5 * epsilon * dtaudp(p.data,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p.data,alpha,lam,Q))
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data.copy_(qprime)
        count = count + 1

    dV,H_,dH = getdH(q,V)
    lam,Q = eigen(H_.data)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    p.data -= 0.5 * dtaudq(p.data,dH,Q,lam,alpha) * epsilon
    p.data -= 0.5 * dphidq(lam,alpha,dH,Q,dV.data) * epsilon
    return(q,p)

def generalized_leapfrog_tensor(q,p,epsilon,alpha,delta,V):
    #
    #lam,Q = eigen(getH(q,V).data)
    #dV,H_ = getH(q,V)
    #dH_i = getdH_specific(q,V,H_)
    #dV = getdV(q,V,True)
    dV,H_,dH = getdH_tensor(q,V)
    lam, Q = eigen(H_)
    p.data -= epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV)
    rho = p.clone()
    pprime = p.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime.copy_(rho - epsilon * 0.5 * dtaudq(p,dH,Q,lam,alpha))
        deltap = torch.max(torch.abs(p-pprime))
        p.copy_(pprime)
        count = count + 1

    sigma = q.clone()
    qprime = q.clone()
    deltaq = delta + 0.5

    _,H_ = getH(sigma,V)
    olam,oQ = eigen(H_)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = getH(q,V)
        lam,Q = eigen(H_)
        qprime.copy_(sigma + 0.5 * epsilon * dtaudp(p,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p,alpha,lam,Q))
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data.copy_(qprime)
        count = count + 1

    dV,H_,dH = getdH(q,V)
    lam,Q = eigen(H_)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    p -= 0.5 * dtaudq(p,dH,Q,lam,alpha) * epsilon
    p -= 0.5 * dphidq(lam,alpha,dH,Q,dV) * epsilon
    return(q,p)
def generalized_leapfrog_specific(q,p,epsilon,alpha,delta,V):
    # here dH is calculated one dimension at a time.
    # O(N^2) memory but repeats derivative calculation 5,6 times-- still O(N^2) time

    #lam,Q = eigen(getH(q,V).data)
    #dV,H_ = getH(q,V)
    dV,H_,dH_i = getdH_specific(q,V)
    #dV = getdV(q,V,True)
    #dV,H_,dH = getdH(q,V)
    lam, Q = eigen(H_.data)
    p.data -= epsilon * 0.5 * dphidq(lam,alpha,dH_i,Q,dV.data)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime = rho - epsilon * 0.5 * dtaudq(p.data,dH_i,Q,lam,alpha)
        deltap = torch.max(torch.abs(p.data-pprime))
        p.data = pprime
        count = count + 1

    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5

    _,H_ = getH(sigma,V)
    olam,oQ = eigen(H_.data)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = getH(q,V)
        lam,Q = eigen(H_.data)
        qprime = sigma.data + 0.5 * epsilon * dtaudp(p.data,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p.data,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data = qprime
        count = count + 1

    dV, H_ = getH(q, V)
    dH_i = getdH_specific(q, V, H_)
    lam,Q = eigen(H_.data)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    p.data -= 0.5 * dtaudq(p.data,dH_i,Q,lam,alpha) * epsilon
    p.data -= 0.5 * dphidq(lam,alpha,dH_i,Q,dV.data) * epsilon
    return(q,p)

def generalized_leapfrog_specific_tensor(q,p,epsilon,alpha,delta,V):
    # here dH is calculated one dimension at a time.
    # O(N^2) memory but repeats derivative calculation 5,6 times-- still O(N^2) time

    #lam,Q = eigen(getH(q,V).data)
    #dV,H_ = getH(q,V)
    dV,H_,dH_i = getdH_specific(q,V)
    #dV = getdV(q,V,True)
    #dV,H_,dH = getdH(q,V)
    lam, Q = eigen(H_)
    p -= epsilon * 0.5 * dphidq(lam,alpha,dH_i,Q,dV)
    rho = p.clone()
    pprime = p.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime = rho - epsilon * 0.5 * dtaudq(p,dH_i,Q,lam,alpha)
        deltap = torch.max(torch.abs(p-pprime))
        p = pprime
        count = count + 1

    sigma = q.clone()
    qprime = q.clone()
    deltaq = delta + 0.5

    _,H_ = getH(sigma,V)
    olam,oQ = eigen(H_)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = getH(q,V)
        lam,Q = eigen(H_.data)
        qprime = sigma.data + 0.5 * epsilon * dtaudp(p,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q-qprime))
        q = qprime
        count = count + 1

    #dV, H_ = getH(q, V)
    dV,H_dH_i = getdH_specific(q, V)
    lam,Q = eigen(H_)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    p -= 0.5 * dtaudq(p,dH_i,Q,lam,alpha) * epsilon
    p -= 0.5 * dphidq(lam,alpha,dH_i,Q,dV.data) * epsilon
    return(q,p)

def generalized_leapfrog_windowed(q,p,epsilon,alpha,delta,V,logw_old,qprop_old,pprop_old):
    #
    #lam,Q = eigen(getH(q,V).data)
    #dV,H_ = getH(q,V)
    #dH_i = getdH_specific(q,V,H_)
    #dV = getdV(q,V,True)
    dV,H_,dH = getdH(q,V)
    lam, Q = eigen(H_.data)
    p.data -= epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV.data)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime.copy_(rho - epsilon * 0.5 * dtaudq(p.data,dH,Q,lam,alpha))
        deltap = torch.max(torch.abs(p.data-pprime))
        p.data.copy_(pprime)
        count = count + 1

    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5

    _,H_ = getH(sigma,V)
    olam,oQ = eigen(H_.data)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = getH(q,V)
        lam,Q = eigen(H_.data)
        qprime.copy_(sigma.data + 0.5 * epsilon * dtaudp(p.data,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p.data,alpha,lam,Q))
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data.copy_(qprime)
        count = count + 1

    dV,H_,dH = getdH(q,V)
    lam,Q = eigen(H_.data)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    p.data -= 0.5 * dtaudq(p.data,dH,Q,lam,alpha) * epsilon
    p.data -= 0.5 * dphidq(lam,alpha,dH,Q,dV.data) * epsilon
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

def generalized_leapfrog_windowed_tensor(q,p,epsilon,alpha,delta,V,T,H,logw_old,qprop_old,pprop_old):
    #
    #lam,Q = eigen(getH(q,V).data)
    #dV,H_ = getH(q,V)
    #dH_i = getdH_specific(q,V,H_)
    #dV = getdV(q,V,True)
    dV,H_,dH = getdH_tensor(q,V)
    lam, Q = eigen(H_)
    p -= epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV)
    rho = p.clone()
    pprime = p.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime.copy_(rho - epsilon * 0.5 * dtaudq(p,dH,Q,lam,alpha))
        deltap = torch.max(torch.abs(p-pprime))
        p.copy_(pprime)
        count = count + 1

    sigma = q.clone()
    qprime = q.clone()
    deltaq = delta + 0.5

    _,H_ = getH_tensor(sigma,V)
    olam,oQ = eigen(H_)
    #olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = getH_tensor(q,V)
        lam,Q = eigen(H_)
        qprime.copy_(sigma + 0.5 * epsilon * dtaudp(p,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p,alpha,lam,Q))
        deltaq = torch.max(torch.abs(q-qprime))
        q.copy_(qprime)
        count = count + 1

    dV,H_,dH = getdH_tensor(q,V)
    lam,Q = eigen(H_)
    #dH = getdH(q,V)
    #dV = getdV(q,V,False)
    #lam,Q = eigen(getH(q,V).data)
    p -= 0.5 * dtaudq(p,dH,Q,lam,alpha) * epsilon
    p -= 0.5 * dphidq(lam,alpha,dH,Q,dV) * epsilon
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
def rmhmc_step(initq,H,epsilon,L,alpha,delta,V):
    q = Variable(initq.data.clone(), requires_grad=True)
    _, H_ = getH(q, V)
    lam, Q = eigen(H_.data)
    #lam,Q = eigen(getH(q,V).data)
    p = Variable(generate_momentum(alpha,lam,Q))
    current_H = H(q,p,alpha)

    for _ in range(L):
        out = generalized_leapfrog(q,p,epsilon,alpha,delta,V)
        q.data = out[0].data
        p.data = out[1].data

    proposed_H = H(q,p,alpha)
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

def rmhmc_step_windowed(initq,H,epsilon,L,alpha,delta,V):
    q = Variable(initq.data.clone(), requires_grad=True)
    _, H_ = getH(q, V)
    lam, Q = eigen(H_.data)
    #lam,Q = eigen(getH(q,V).data)
    p = Variable(generate_momentum(alpha,lam,Q))


    logw_prop = -H(q, p)
    q_prop = q.clone()
    p_prop = p.clone()
    accep_rate_sum = 0
    num_divergent = 0
    for _ in range(L):
        o = generalized_leapfrog_windowed(q, p, epsilon, V, H, logw_prop, q_prop, p_prop)
        q, p = o[0], o[1]
        q_prop, p_prop = o[2], o[3]
        diff = abs(logw_prop - o[4])
        if abs(diff) > 1000:
            num_divergent += 1
        logw_prop = o[4]

        accep_rate_sum += o[5]

    return(q_prop, accep_rate_sum / L, num_divergent)

def rmhmc_step_tensor(initq,epsilon,L,alpha,delta,V,T,H):
    q = initq.clone()
    _, H_ = getH_tensor(q, V)
    lam, Q = eigen(H_)
    #lam,Q = eigen(getH(q,V).data)
    p = T.generate_momentum(q)
    current_H = H(q,p)

    for _ in range(L):
        out = generalized_leapfrog(q,p,epsilon,alpha,delta,V)
        q = out[0]
        p = out[1]

    proposed_H = H(q,p,alpha)
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

def rmhmc_step_windowed_tensor(initq,V,T,H,epsilon,L,alpha,delta):
    q = initq.clone()
    _, H_ = getH_tensor(q, V)
    lam, Q = eigen(H_)
    #lam,Q = eigen(getH(q,V).data)
    p = T.generate_momentum(alpha,lam,Q)


    logw_prop = -H(q, p)
    q_prop = q.clone()
    p_prop = p.clone()
    accep_rate_sum = 0
    num_divergent = 0
    for _ in range(L):
        o = generalized_leapfrog_windowed_tensor(q, p, epsilon, V, H, logw_prop, q_prop, p_prop)
        q, p = o[0], o[1]
        q_prop, p_prop = o[2], o[3]
        diff = abs(logw_prop - o[4])
        if abs(diff) > 1000:
            num_divergent += 1
        logw_prop = o[4]

        accep_rate_sum += o[5]

    return(q_prop, accep_rate_sum / L, num_divergent)