import torch
from torch.autograd import Variable,grad
import numpy
def pi(q,p):
    H = torch.dot(q,q) * 0.5 + torch.dot(p,p) * 0.5
    return(H)
def leapfrog(q,p,epsilon,pi):
    p_prime = Variable(p.data.clone(),requires_grad=False)
    q_prime = Variable(q.data.clone(),requires_grad=True)
    #print(q_prime.data,p_prime.data)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5 * epsilon
    #print(q_prime.data,p_prime.data)
    q_prime.grad.data.zero_()
    q_prime.data += epsilon * p_prime.data
    #print(q_prime.data,p_prime.data)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5 * epsilon
    #print(q_prime.data,p_prime.data)
    q_prime.grad.data.zero_()
    return(q_prime,p_prime)

def HMC_alt(epsilon, L, current_q, leapfrog, pi):
    p = Variable(torch.randn(len(current_q)), requires_grad=False)
    q = Variable(current_q.data.clone(), requires_grad=True)
    #print("original q,p are {},{}".format(q,p))
    current_H = pi(q, p)
    for _ in range(L):
        temp_q, temp_p = leapfrog(q, p, epsilon, pi)
        q.data, p.data = temp_q.data.clone(), temp_p.data.clone()

    #print("propsed q, p are {},{}".format(q,p))
    proposed_H = pi(q, p)
    temp = torch.exp(current_H - proposed_H)

    #print("current H is {}".format(current_H))
    #print("proposed H is {}".format(proposed_H))
    #print("temp is {}".format(temp))
    if (numpy.random.random(1) < temp.data.numpy()):
        return (q)
    else:
        return (current_q)


dim = 2
q = Variable(torch.randn(dim),requires_grad=True)
#p = Variable(torch.randn(dim),requires_grad=True)
#print("q is {}".format(q.data))
#print("torch output {}".format(torch.dot(q.data,q.data)))
#print(numpy.dot(q.data.numpy(),q.data.numpy()))
#print("p is {}".format(p.data))
#print("torch.output{}".format(torch.dot(p.data,p.data)))
#print(numpy.dot(p.data.numpy(),p.data.numpy()))
#print("H is {}".format(pi(q,p).data.numpy()))
#exit()
chain_l = 2000
burn_in = 1000
store = torch.zeros((chain_l,dim))
for i in range(chain_l):
    #print("round {}".format(i))
    #out = HMC(0.1,10,q)
    out = HMC_alt(0.1,10,q,leapfrog,pi)
    #out = NUTS(q,0.1,pi,leapfrog,NUTS_criterion)
    #print("tree depth is {}".format(out[1]))
    #store[i,] = out[0].data
    store[i,]=out.data
    q.data = out.data
    #q.data = out[0].data
    #print("q is {}".format(q.data))

store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
print("length of chain is {}".format(chain_l))
print("burn in is {}".format(burn_in))
print(empCov)
print(emmean)

