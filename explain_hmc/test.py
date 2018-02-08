from likelihood import likelihood
from hmc import hmc_sampler
import sys
import numpy
import torch
from torch.autograd import Variable,grad
dim = 3
q = Variable(torch.rand(dim),requires_grad=True)
SigInv = Variable(torch.eye(dim),requires_grad=False)
potentialE =  q.dot(SigInv.mv(q*q))
#print(q.data)
g = grad(potentialE,q,create_graph=True)[0]
#print(g)
gsplit = torch.split(g,1,dim=0)
#print(gsplit)
H = Variable(torch.rand(dim,dim))
for i in range(dim):
    H[i,:] = grad(gsplit[i],q,create_graph=True)[0]
#print(H)
dH = Variable(torch.rand(dim,dim,dim))
print(H)
#exit()
#x = Variable(torch.rand(1),requires_grad=True)
#y = 0.5 *x
#o = grad(y,x,create_graph=True)
#print(o)
#oo = grad(Variable(torch.rand(1),requires_grad=True),x)
#o = grad(H[0,0],q)
#print(o)
for i in range(dim):
    for j in range(dim):
        dH[i,j,:] = grad(H[i,j],q,create_graph=True)[0]

#print(dH[0,0,])
#potentialE.backward(retain_graph=True,create_graph=True)
#V = q.grad
#o=torch.autograd.grad(potentialE,q,retain_graph=True,create_graph=True)
#V = o[0]
#print(V[0])
#exit()
#oo = torch.autograd.backward(V[0],q,retain_graph=True,create_graph=True)

from torch.autograd import Variable, grad, backward
import torch

x = Variable(torch.ones(1), requires_grad=True)
y = x.pow(3)
#gradient = torch.randn(2)
#print(y[0])
#print(torch.split(y,1,dim=0))
#g = grad(torch.split(y,1,dim=0),inputs=[x,x])
#print(g) # g = 3
#x = Variable(torch.ones(2),requires_grad=True)
#y = x + 2
#z = y * y
#o=grad()
#print(x.grad)
#g2 = grad(g[0], x)
#print(g2) # g2 = 6

import numpy
import torch
from torch.autograd import Variable

dim = 3
chain_l = 500
q = Variable(torch.rand(dim))
store = torch.rand((chain_l,dim))

def pi(q,p):
    H = torch.dot(q,q) * 0.5 + torch.dot(p,p) * 0.5
    return(H)
def pi_numpy(q,p):
    H = numpy.dot(q,q) * 0.5 + numpy.dot(p,p) * 0.5
    return(H)
def leapfrog_explicit(q,p,epsilon,pi):
    p_prime = Variable(p.data.clone())
    q_prime = Variable(q.data.clone())
    p_prime.data -= q_prime.data * 0.5 * epsilon
    q_prime.data += epsilon * p_prime.data
    p_prime.data -=q_prime.data * 0.5 * epsilon
    return(q_prime.data,p_prime.data)
def leapfrog_numpy(q,p,epsilon):
    p_prime = numpy.copy(p)
    q_prime = numpy.copy(q)
    p_prime -= q_prime * 0.5 * epsilon
    q_prime += epsilon * p_prime
    p_prime -= q_prime * 0.5 * epsilon
    return(q_prime,p_prime)
def HMC(epsilon,L,current_q,leapfrog,pi):
    p = Variable(torch.randn(len(current_q)))
    q = Variable(current_q.data.clone())
    current_H = pi(q,p)
    for i in range(L):
        q.data,p.data = leapfrog(q,p,epsilon,pi)

    proposed_H = pi(q,p)
    temp = torch.exp(current_H - proposed_H)

    if(numpy.random.random(1) < temp.data.numpy()):
        return(q)
    else:
        return(current_q)

def HMC_numpy(epsilon,L,current_q,leapfrog,pi):
    p = numpy.random.randn(len(current_q))
    q = numpy.copy(current_q)
    current_H = pi_numpy(q,p)
    for i in range(L):
        q,p = leapfrog_numpy(q,p,epsilon)
    proposed_H = pi_numpy(q,p)
    temp = numpy.exp(current_H - proposed_H)
    print("current H {}".format(current_H))
    print("proposed H {}".format(proposed_H))
    if(numpy.random.random(1)<temp):
        return(q)
    else:
        return(current_q)


#q = numpy.random.randn(dim)
#p = numpy.random.randn(dim)
#print(q,p)
#for i in range(10):
#    q,p = leapfrog_numpy(q,p,0.1)
#print(q,p)
#exit()
#store = numpy.zeros((chain_l,dim))
print(q)
#out = HMC(0.1,10,q,leapfrog_explicit,pi)
#print(out)
#exit()
for i in range(chain_l):
    print("round {}".format(i))
    out = HMC(0.1,10,q,leapfrog_explicit,pi)
    store[i,]=out.data.clone()
    #q = out.copy()
    q.data = out.data.clone()
#o=numpy.cov(store)
#print(o)
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
print(empCov)
emmean = numpy.mean(store,axis=0)
print(emmean)
#print(store[1:20,4])