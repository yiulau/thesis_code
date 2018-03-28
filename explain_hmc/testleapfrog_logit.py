import pandas as pd
import torch
from torch.autograd import Variable,grad
import pystan
import numpy
import pickle
import pandas as pd
dim = 4
num_ob = 25
chain_l = 2000
burn_in = 1000
recompile = False
if recompile:
    mod = pystan.StanModel(file="./alt_log_reg.stan")
    with open('model.pkl', 'wb') as f:
        pickle.dump(mod, f)

mod = pickle.load(open('model.pkl', 'rb'))


y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
X_np = numpy.random.randn(num_ob,dim)

df = pd.read_csv("./pima_india.csv",header=0,sep=" ")
#print(df)
dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
y_np = dfm[:,8]
y_np = y_np.astype(numpy.int64)
X_np = dfm[:,1:8]
dim = X_np.shape[1]
num_ob = X_np.shape[0]
data = dict(y=y_np,X=X_np,N=num_ob,p=dim)
fit = mod.sampling(data=data,refresh=0)
#print(fit)
#exit()

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))

def logsumexp(a,b):
    # input torch vector
    s = torch.max(a,b)
    out = s + torch.log(torch.exp(a-s) + torch.exp(b-s))
    return(out)
def pi(q,p):
    beta = q
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(torch.log(1 + torch.exp(torch.mv(X, beta))))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood

    momentum = torch.dot(p,p) * 0.5

    return(-posterior + momentum)

def pi(q,p):
    beta = q
    #print("xbeta {}".format((torch.mv(X, beta))))
    term1 = torch.sum(logsumexp(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(logsumexp(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood

    momentum = torch.dot(p,p) * 0.5

    return(-posterior + momentum)

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
    #print(q_prime,p_prime)

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
    #exit()
    proposed_H = pi(q, p)
    temp = (current_H - proposed_H)

    #print("current H is {}".format(current_H))
    #print("proposed H is {}".format(proposed_H))

    #print("temp is {}".format(temp))
    if (numpy.log(numpy.random.random(1)) < min(0,temp.data.numpy())):
        return (q)
    else:
        return (current_q)

def HMC_alt_jitter(epsilon, time, current_q, leapfrog, pi):
    p = Variable(torch.randn(len(current_q)), requires_grad=False)
    q = Variable(current_q.data.clone(), requires_grad=True)
    # print("original q,p are {},{}".format(q,p))
    current_H = pi(q, p)
    epsilon = epsilon * (1 + numpy.random.uniform(-0.9,0.9))
    L = max(1,round(time/epsilon))
    print("ep is {}, L is {}".format(epsilon,L))
    for _ in range(L):
        temp_q, temp_p = leapfrog(q, p, epsilon, pi)
        q.data, p.data = temp_q.data.clone(), temp_p.data.clone()

    # print("propsed q, p are {},{}".format(q,p))
    # exit()
    proposed_H = pi(q, p)
    temp = (current_H - proposed_H)

    # print("current H is {}".format(current_H))
    # print("proposed H is {}".format(proposed_H))

    # print("temp is {}".format(temp))
    if (numpy.log(numpy.random.random(1)) < min(0, temp.data.numpy())):
        return (q)
    else:
        return (current_q)

#p = Variable(torch.randn(dim),requires_grad=True)
#out = HMC_alt(0.1,10,q,leapfrog,pi)
#print(out)
#exit()
#print("q is {}".format(q.data))
#print("torch output {}".format(torch.dot(q.data,q.data)))
#print(numpy.dot(q.data.numpy(),q.data.numpy()))
#print("p is {}".format(p.data)) K #print(pi(q,p))
#exit()
store = torch.zeros((chain_l,dim))
for i in range(chain_l):
    print("round {}".format(i))
    #out = HMC(0.1,10,q)
    #out = HMC_alt(0.12,10,q,leapfrog,pi)
    out = HMC_alt_jitter(0.1,1,q,leapfrog,pi)
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
#print(empCov)
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))


print(fit)

