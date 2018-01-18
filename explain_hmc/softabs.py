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















