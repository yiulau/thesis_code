import numpy as np
import torch
from math import exp
def coth_torch(x):
    return((torch.exp(x) + torch.exp(-x))/(torch.exp(x)-torch.exp(-x)))
def coth(x):
    return((exp(x)+exp(-x))/exp(x)-exp(x))
def softabs_map(lam,alpha):
    # takes vector as input
    # returns a vector
    return(((torch.exp(alpha*lam) + torch.exp(-alpha*lam))/
        torch.exp(alpha*lam) - torch.exp(-alpha*lam)) * lam)

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

def getH(q):
    length = len(q)
    input = torch.rand(length,length)
    input = torch.mm(input,torch.t(input))
    return(input)
def getdH(q):
    length = len(q)
    return(torch.rand(length,length,length))
def getdV(q):
    length = len(q)
    return(torch.rand(length))
def generalized_leapfrog(q,p,epsilon,alpha,delta):
    lam,Q = eigen(getH(q))
    dH = getdH(q)
    dV = getdV(q)
    #p = generate_momentum(alpha,lam,Q)
    p = p - epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV)
    rho = p.clone()
    pprime = p.clone()
    deltap = delta + 0.5
    while deltap > delta:
        pprime = rho - epsilon * 0.5 * dtaudq(p,dH,Q,lam,alpha)
        deltap = torch.max(torch.abs(p-pprime))
        p = pprime.clone()
    sigma = q.clone()
    qprime = q.clone()
    deltaq = delta + 0.5
    olam,oQ = eigen(getH(sigma))
    while deltaq > delta:
        lam,Q = eigen(getH(q))
        qprime = sigma + 0.5 * dtaudp(p,alpha,olam,oQ) + 0.5 * dtaudp(p,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q-qprime))
        q = qprime.clone()
    dH = getdH(q)
    dV = getdV(q)
    lam,Q = eigen(getH(q))
    p = p - 0.5 * dtaudq(p,dH,Q,lam,alpha)
    p = p - 0.5 * dphidq(lam,alpha,dH,Q,dV)
    return(q,p)




def rmhmc_step(initq,pi):



    return(0)