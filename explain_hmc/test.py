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
print(g)
gsplit = torch.split(g,1,dim=0)
print(gsplit)
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

print(dH[0,0,])
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