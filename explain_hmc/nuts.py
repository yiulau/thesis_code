import numpy
import torch
from torch.autograd import Variable
def NUTS(q_init,epsilon,pi,leapfrog,NUTS_criterion):
    p = Variable(torch.randn(len(q_init)),requires_grad=True)
    q_left = q_init.clone()
    q_right = q_init.clone()
    p_left = p.clone()
    p_right = p.clone()
    j = 0
    q_prop = q_init.clone()
    w = torch.exp(-pi(q_init,p))
    s = True
    while s:
        v = numpy.asscalar((numpy.random.randn(1)>0)*2 -1)
        if v < 0:
            q_left,p_left,_,_,q_prime,s_prime,w_prime = BuildTree(q_left,p_left,v,j,epsilon,leapfrog,pi,NUTS_criterion)
        else:
            _,_,q_right,p_right,q_prime,s_prime,w_prime = BuildTree(q_right,p_right,v,j,epsilon,leapfrog,pi,NUTS_criterion)
        if s_prime:
            accep_rate = min(1,(w_prime/w).data.numpy())
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prop = q_prime
            w = w + w_prime
            s = s and NUTS_criterion(q_left,q_right,p_left,p_right)
            j = j + 1
    return(q_prop)

def BuildTree(q,p,v,j,epsilon,leapfrog,pi,NUTS_criterion):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,pi)
        w_prime = torch.exp(-pi(q_prime,p_prime))
        return q_prime,p_prime,q_prime,p_prime,q_prime,True,w_prime
    else:
        q_left,p_left,q_right,p_right,q_prime,s_prime,w_prime = BuildTree(q,p,v,j-1,epsilon,leapfrog,pi,NUTS_criterion)
        if s_prime:
            if v ==-1:
                q_left,p_left,_,_,q_dprime,s_dprime,w_dprime = BuildTree(q_left,p_left,v,j-1,epsilon,leapfrog,pi,NUTS_criterion)
            else:
                _,_,q_right,p_right,q_dprime,s_dprime,w_dprime = BuildTree(q_right,p_right,v,j-1,epsilon,leapfrog,pi,NUTS_criterion)
            accep_rate = min(1,(w_dprime/(w_prime+w_dprime)).data.numpy())
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prime = q_dprime.clone()
            s_prime = s_dprime and NUTS_criterion(q_left,q_right,p_left,p_right)
            w_prime = w_prime + w_dprime
        return q_left,p_left,p_left,p_right,q_prime,s_prime,w_prime

def leapfrog(q,p,epsilon,pi):
    p_prime = Variable(p.data.clone(),requires_grad=True)
    q_prime = Variable(q.data.clone(),requires_grad=True)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data = p_prime.data + q_prime.grad.data * 0.5
    q_prime.grad.data.zero_()
    q_prime.data = q_prime.data + epsilon * p_prime.data
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data = p_prime.data + q_prime.grad.data * 0.5
    return(q_prime,p_prime)

def pi(q,p):
    H = torch.dot(q,q) * 0.5 + torch.dot(p,p) * 0.5
    return(H)

def NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue going
    # False = stops
    o = (torch.dot(q_right-q_left,p_right).data.numpy() >=0)[0] and \
        (torch.dot(q_right-q_left,p_left).data.numpy() >=0)[0]
    return(o)



#H = pi(q_prime, p_prime)
#H = torch.dot(q,q) * 0.5 + torch.dot(p,p) * 0.5
#H.backward()
#print(q.grad)
##p_prime.data = p_prime.data + q_prime.grad.data * 0.5
#exit()
q = Variable(torch.randn(2),requires_grad=True)
p = Variable(torch.randn(2),requires_grad=True)
q.data = q.data + 0.1 * p.data
v=(numpy.random.randn(1)>0)*2 -1
#print(type(numpy.asscalar(v)))
#print(p)
##print(v)
#print(numpy.asscalar(v)*p)

#print((torch.dot(q,p).data.numpy()>=0)[0])

#print(NUTS_criterion(q+1,q,p+1,p))

#print(BuildTree(q,p,-1,1,0.1,leapfrog,pi,NUTS_criterion))

print(NUTS(q,0.1,pi,leapfrog,NUTS_criterion))