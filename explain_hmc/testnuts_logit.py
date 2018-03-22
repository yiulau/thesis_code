import pandas as pd
import torch
from torch.autograd import Variable,grad
import pystan
import numpy
import pickle
import time, cProfile, math
dim = 5
num_ob = 100
chain_l = 500
burn_in = 100
max_tdepth = 10

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

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))
def pi(q,p):
    beta = q
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(torch.log(1 + torch.exp(torch.mv(X, beta))))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood

    momentum = torch.dot(p,p) * 0.5

    return(-posterior + momentum)

def logsumexp(a, b):
    s = max(a,b)
    output = s + math.log((math.exp(a-s) + math.exp(b-s)))
    return(output)

def NUTS(q_init,epsilon,pi,leapfrog,NUTS_criterion):
    p = Variable(torch.randn(len(q_init)),requires_grad=False)
    q_left = Variable(q_init.data.clone(),requires_grad=True)
    q_right = Variable(q_init.data.clone(),requires_grad=True)
    p_left = Variable(p.data.clone(),requires_grad=False)
    p_right = Variable(p.data.clone(),requires_grad=False)
    j = 0
    q_prop = Variable(q_init.data.clone(),requires_grad=True)
    #log_w = -pi(q_init.data,p.data)
    log_w = -pi(q_init,p).data.numpy()
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, s_prime, log_w_prime = BuildTree(q_left, p_left, -1, j, epsilon, leapfrog, pi,
                                                                            NUTS_criterion)
        else:
            _, _, q_right, p_right, q_prime, s_prime, log_w_prime = BuildTree(q_right, p_right, 1, j, epsilon, leapfrog, pi,
                                                                              NUTS_criterion)
        if s_prime:
            accep_rate = numpy.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prop.data = q_prime.data.clone()
        log_w = logsumexp(log_w,log_w_prime)
        s = s_prime and NUTS_criterion(q_left,q_right,p_left,p_right)
        j = j + 1
        s = s and (j<max_tdepth)
    return(q_prop,j)

def BuildTree(q,p,v,j,epsilon,leapfrog,pi,NUTS_criterion):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,pi)
        #log_w_prime = -pi(q_prime.data,p_prime.data)
        log_w_prime = -pi(q_prime, p_prime).data.numpy()
        return q_prime, p_prime, q_prime, p_prime, q_prime, True, log_w_prime
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime = BuildTree(q, p, v, j - 1, epsilon, leapfrog, pi, NUTS_criterion)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,s_dprime,log_w_dprime = BuildTree(q_left,p_left,v,j-1,epsilon,leapfrog,pi,NUTS_criterion)
            else:
                _, _, q_right, p_right, q_dprime, s_dprime, log_w_dprime = BuildTree(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, pi, NUTS_criterion)
            accep_rate = numpy.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accep_rate:
                q_prime.data = q_dprime.data.clone()
            s_prime = s_dprime and NUTS_criterion(q_left,q_right,p_left,p_right)
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime

def leapfrog(q,p,epsilon,pi):
    p_prime = Variable(p.data.clone(),requires_grad=False)
    q_prime = Variable(q.data.clone(),requires_grad=True)
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5 * epsilon
    q_prime.grad.data.zero_()
    q_prime.data += epsilon * p_prime.data
    H = pi(q_prime,p_prime)
    H.backward()
    p_prime.data -= q_prime.grad.data * 0.5 * epsilon
    q_prime.grad.data.zero_()
    return(q_prime, p_prime)



def NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue going
    # False = stops
    o = (torch.dot(q_right.data-q_left.data,p_right.data) >=0) or \
        (torch.dot(q_right.data-q_left.data,p_left.data) >=0)
    return(o)

#q = Variable(torch.randn(dim),requires_grad=True)

v = -1

epsilon = 0.11

store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    #print("round {}".format(i))
    out = NUTS(q,0.12,pi,leapfrog,NUTS_criterion)
    store[i,] = out[0].data # turn this on when using Nuts
    q.data = out[0].data # turn this on when using nuts

total = time.time() - begin
print("total time is {}".format(total))
print("length of chain is {}".format(chain_l))
print("length of burn in is {}".format(burn_in))
print("Use logit")
store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
#print(empCov)
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))
print(fit)