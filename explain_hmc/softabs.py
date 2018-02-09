import numpy as np
import torch
from torch.autograd import Variable,grad
from math import exp
def coth_torch(x):
    return(1/torch.tanh(x))
def coth(x):
    return(1/np.asscalar(np.tanh(x)))
def softabs_map(lam,alpha):
    # takes vector as input
    # returns a vector
    return(1/torch.tanh(lam*alpha))
def J(lam,alpha,length):
    J = torch.zeros(length,length)
    for i in range(length):
        for j in range(length):
            J[i,j] = (lam[i]*coth(alpha*lam[i]) - lam[j]*coth(alpha*lam[j]))*(1.-1*(i==j))+\
                     (coth(alpha*lam[i]) + lam[i]*(1-coth(alpha*lam[i]))*alpha)*(1*(i==j))
    return(J)

def D(p,Q,lam,alpha):
    return(torch.diag(torch.mv(torch.t(Q),p)/(lam*coth_torch(alpha*lam))))
def dtaudp(p,alpha,lam,Q):
    return(torch.mv(Q,(softabs_map(lam,alpha)*torch.mv(torch.t(Q),p))))



def dtaudq(p,dH,Q,lam,alpha):
    N = len(p)
    Jm = J(lam,alpha,len(p))
    Dm = D(p,Q,lam,alpha)
    M = torch.mm(Q,torch.mm(Dm,torch.mm(Jm,torch.mm(Dm,torch.t(Q)))))

    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(-torch.mm(M,dH[i,:,:]))
    return(delta)


def dphidq(lam,alpha,dH,Q,dV):
    N = len(lam)
    Jm = J(lam,alpha,len(lam))
    R = torch.diag(1/(lam*coth_torch(alpha*lam)))
    M = torch.mm(Q,torch.mm(R*Jm,torch.t(Q)))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(torch.mm(M,dH[i,:,:])) + dV[i]
    return(delta)


def eigen(H):
    # input must be of type tensor ** not variable
    out = torch.eig(H,True)
    return(out[0][:,0],out[1])

input = torch.rand(2,2)
input = torch.mm(input,torch.t(input))
out = eigen(input)
lam = out[0]
Q = out[1]
#print(input)
#print(torch.mm(Q,torch.mm(torch.diag(lam),torch.t(Q))))
# generate_momentum need ot be tested for correctness
def generate_momentum(alpha,lam,Q):
    # generate by multiplying st normal by QV^(0.5) where Sig = QVQ^T
    temp = torch.mm(Q,torch.diag(1./torch.sqrt(softabs_map(lam,alpha))))
    out = torch.mv(temp,torch.randn(len(lam)))
    return(out)

def getdV(q,V):
    potentialE = V(q)
    g = grad(potentialE, q, create_graph=True)[0]
    return(g)

def getH(q,V):
    g = getdV(q,V)
    dim = len(q)
    H = Variable(torch.zeros(dim, dim))
    for i in range(dim):
        H[i, :] = grad(g[i], q, create_graph=True)[0]

    return(H)

def getdH(q,V):
    H = getH(q,V)
    dim = len(q)
    dH = Variable(torch.zeros(dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            try:
                dH[i, j, :] = grad(H[i, j], q, create_graph=True)[0]
            except RuntimeError:
                dH[i, j, :] = Variable(torch.zeros(len(q)))

    return(dH)

def V(q):
    # returns variable if q is variable , returns float if q tensor (shouldn't need it tho)
    return(0.5 * torch.dot(q,q))

def T(q,p,alpha):
    H = getH(q,V)
    out = eigen(H.data)
    lam = out[0]
    Q = out[1]
    temp = softabs_map(lam,alpha)
    inv_exp_H = torch.mm(torch.mm(Q,torch.diag(temp)),torch.t(Q))
    o = 0.5 * torch.dot(p.data,torch.mv(inv_exp_H,p.data))
    return(o)

def H(q,p,alpha):
    return(V(q).data[0] + T(q,p,alpha))
def generalized_leapfrog(q,p,epsilon,alpha,delta,V):
    lam,Q = eigen(getH(q,V).data)
    dH = getdH(q,V)
    dV = getdV(q,V)
    #p = generate_momentum(alpha,lam,Q)
    p.data = p.data - epsilon * 0.5 * dphidq(lam,alpha,dH.data,Q,dV.data)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    while deltap > delta:
        pprime = rho - epsilon * 0.5 * dtaudq(p.data,dH.data,Q,lam,alpha)
        deltap = torch.max(torch.abs(p.data-pprime))
        p.data = pprime.clone()
    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5
    olam,oQ = eigen(getH(sigma,V).data)
    while deltaq > delta:
        lam,Q = eigen(getH(q,V).data)
        qprime = sigma.data + 0.5 * dtaudp(p.data,alpha,olam,oQ) + 0.5 * dtaudp(p.data,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data = qprime.clone()
    dH = getdH(q,V)
    dV = getdV(q,V)
    lam,Q = eigen(getH(q,V).data)
    p.data = p.data - 0.5 * dtaudq(p.data,dH.data,Q,lam,alpha)
    p.data = p.data - 0.5 * dphidq(lam,alpha,dH.data,Q,dV.data)
    return(q,p,H(q,p,alpha))




def rmhmc_step(initq,H,epsilon,alpha,delta):
    p = Variable(torch.randn(len(initq)),requires_grad=True)
    q = Variable(initq.data,requires_grad=True)
    current_H = H(q,p,alpha)
    out = generalized_leapfrog(q,p,epsilon,alpha,delta)
    proposed_H = out[2]
    u = np.random.rand(1)
    if u < np.exp(current_H - proposed_H):
        return(out[0])
    else:
        return(q)

q = Variable(torch.randn(2),requires_grad=True)
p =Variable(torch.randn(2),requires_grad=True)

inv_exp_H = T(p,q,50)
ou = generalized_leapfrog(q,p,0.1,50,0.1,V)
print(ou)
exit()
#print(Q)
#print(torch.mm(Q,torch.diag(temp)))
#exit()
print(T(q,p,50))
#print(0.5*torch.dot(p,p))
#exit()
print(q)
out = getdV(q,V)
out2 = getH(q,V)
#out3 = getdH(q,V)
#print(out)
print(out2)
#print(out3[1,:,:])


