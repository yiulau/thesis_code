import torch, numpy, math
from torch.autograd import Variable,grad

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
    upper_softabs_thresh = 10000
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
    g = grad(V(q), q, create_graph=want_graph)[0]
    return(g)

def getH(q,V):
    # output: H - Pytorch Variable
    g = getdV(q,V,True)
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        H = Variable(torch.zeros(dim, dim)).cuda()
    else:
        H = Variable(torch.zeros(dim, dim))
    for i in range(dim):
        H[i, :] = grad(g[i], q, create_graph=True)[0]
    return(H)

def getdH(q,V):
    # output: dH - pytorch tensor where
    H = getH(q,V)
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        dH = torch.zeros(dim,dim,dim).cuda()
    else:
        dH = torch.zeros(dim,dim,dim)
    for i in range(dim):
        for j in range(dim):
            dH[:, i, j] = grad(H[i, j], q, create_graph=False,retain_graph=True)[0].data

    return(dH)

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
                alp_lam = lam[i] * alpha
                if alp_lam < lower_softabs_thresh:
                    J[i,j] = (2./3.) * alp_lam * (1.- (2./15.)*alp_lam*alp_lam)
                elif alp_lam > upper_softabs_thresh:
                    # 1 if lam > 0 , -1 otherwise
                    J[i,j] = 2*float(lam[i]>0)-1
                else:
                    J[i, j] = (coth(alpha * lam[i]) + lam[i] * (1 - (coth(alpha * lam[i]))**2) * alpha)
            else:
                J[i,j] = (lam[i]*coth(alpha*lam[i]) - lam[j]*coth(alpha*lam[j]))/(lam[i]-lam[j])
            j = j + 1
        i = i + 1
    return(J)

def D(p,Q,lam,alpha):
    # output : Diag( (Q^T * p) / lam * coth(alpha * lam ))
    # Diagonal matrix such that ith entry is (Q^T * p)_i / ( lam_i * coth(alpha * lam_i))
    return(torch.diag(torch.mv(torch.t(Q),p)/(lam*coth_torch(alpha*lam))))

def dtaudq(p,dH,Q,lam,alpha):
    N = len(p)
    Jm = J(lam,alpha,len(p))
    Dm = D(p,Q,lam,alpha)
    M = torch.mm(Q,torch.mm(Dm,torch.mm(Jm,torch.mm(Dm,torch.t(Q)))))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(-torch.mm(M,dH[i,:,:]))
    return (delta)

def dtaudp(p,alpha,lam,Q):
    return(Q.mv(torch.diag(1/softabs_map(lam,alpha)).mv((torch.t(Q).mv(p)))))


def generalized_leapfrog(q,p,epsilon,alpha,delta,V):
    #
    lam,Q = eigen(getH(q,V).data)
    dH = getdH(q,V)
    dV = getdV(q,V,True)
    p.data = p.data - epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV.data)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime = rho - epsilon * 0.5 * dtaudq(p.data,dH,Q,lam,alpha)
        deltap = torch.max(torch.abs(p.data-pprime))
        p.data = pprime
        count = count + 1

    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5
    olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):
        lam,Q = eigen(getH(q,V).data)
        qprime = sigma.data + 0.5 * epsilon * dtaudp(p.data,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p.data,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data = qprime
        count = count + 1

    dH = getdH(q,V)
    dV = getdV(q,V,False)
    lam,Q = eigen(getH(q,V).data)
    p.data = p.data - 0.5 * dtaudq(p.data,dH,Q,lam,alpha) * epsilon
    p.data = p.data - 0.5 * dphidq(lam,alpha,dH,Q,dV.data) * epsilon
    return(q,p)

def rmhmc_step(initq,H,epsilon,L,alpha,delta,V):
    q = Variable(initq.data.clone(), requires_grad=True)
    lam,Q = eigen(getH(q,V).data)
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