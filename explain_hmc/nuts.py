import numpy
import torch
from torch.autograd import Variable
def NUTS(q_init,epsilon,pi,leapfrog,NUTS_criterion):
    p = Variable(torch.randn(len(q_init)),requires_grad=True)
    q_left = Variable(q_init.data.clone(),requires_grad=True)
    q_right = Variable(q_init.data.clone(),requires_grad=True)
    p_left = Variable(p.data.clone(),requires_grad=True)
    p_right = Variable(p.data.clone(),requires_grad=True)
    j = 0
    q_prop = Variable(q_init.data.clone(),requires_grad=True)
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
            s = s and (j<6)
    return(q_prop,j)

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
    p_prime = Variable(p.data.clone(),requires_grad=False)
    q_prime = Variable(q.data.clone(),requires_grad=True)
    #print(q_prime.data,p_prime.data)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5
    #print(q_prime.data,p_prime.data)
    q_prime.grad.data.zero_()
    q_prime.data += epsilon * p_prime.data
    #print(q_prime.data,p_prime.data)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5
    #print(q_prime.data,p_prime.data)
    q_prime.grad.data.zero_()
    return(q_prime,p_prime)

def leapfrog_explicit(q,p,epsilon,pi):
    p_prime = Variable(p.data.clone())
    q_prime = Variable(q.data.clone())
    #print(q_prime.data,p_prime.data)
    p_prime.data -= q_prime.data * 0.5
    #print(q_prime.data,p_prime.data)
    q_prime.data += epsilon * p_prime.data
    #print(q_prime.data,p_prime.data)
    p_prime.data -=q_prime.data * 0.5
    #print(q_prime.data,p_prime.data)
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


def HMC_alt(epsilon, L, current_q, leapfrog, pi):
    p = Variable(torch.randn(len(current_q)), requires_grad=True)
    q = Variable(current_q.data.clone(), requires_grad=True)
    current_H = pi(q, p)
    for _ in range(L):
        temp_q, temp_p = leapfrog(q, p, epsilon, pi)
        q.data, p.data = temp_q.data.clone(), temp_p.data.clone()

    proposed_H = pi(q, p)
    temp = torch.exp(current_H - proposed_H)

    print("current H is {}".format(current_H))
    print("proposed H is {}".format(proposed_H))
    print("temp is {}".format(temp))
    if (numpy.random.random(1) < temp.data.numpy()):
        return (q)
    else:
        return (current_q)


#H = pi(q_prime, p_prime)
#H = torch.dot(q,q) * 0.5 + torch.dot(p,p) * 0.5
#H.backward()
#print(q.grad)
##p_prime.data = p_prime.data + q_prime.grad.data * 0.5
#exit()
q = Variable(torch.randn(2),requires_grad=True)
p = Variable(torch.randn(2),requires_grad=True)
#print("Leapfrog- inside")
#out1 = leapfrog(q,p,0.1,pi)

#print("leapfrog_explicit-inside")
#out2 = leapfrog_explicit(q,p,0.1)

#print("output of leapfrog is {}".format(out1))
#print("output of leapfrog_explicity is {}".format(out2))
#exit()
chain_l = 500
store = torch.zeros((chain_l,2))

for i in range(chain_l):
    print("round {}".format(i))
    #out = HMC(0.1,10,q)
    out = HMC_alt(0.1,10,q,leapfrog,pi)
    store[i,]=out.data
    q.data = out.data

store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
print(empCov)
print(emmean)
exit()
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
chain_l = 200
store = torch.zeros((chain_l,2))
for i in range(chain_l):
    out = NUTS(q,0.2,pi,leapfrog,NUTS_criterion)
    store[i,:] = out[0].data
    q.data = store[i,:]
    print(q.data)
    print("Round {} depth is {}".format(i,out[1]))

store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)

print(empCov)
print(emmean)