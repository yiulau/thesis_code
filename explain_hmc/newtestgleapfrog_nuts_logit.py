import pandas as pd
import torch
from torch.autograd import Variable,grad
import pystan
import numpy
import pickle
import time, cProfile, math
from genleapfrog_ult_util import getH, getdH, getdV, eigen, softabs_map, dphidq, dtaudp, dtaudq, generate_momentum,generalized_leapfrog
from nuts_util import GNUTS
from general_util import logsumexp_torch

dim = 5
num_ob = 100
chain_l = 50
burn_in = 10
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
#fit = mod.sampling(data=data,refresh=0)

#print(fit)

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))



def V(beta):
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(logsumexp_torch(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood
    return(-posterior)

def T(q,alpha):
    def T_givenq(p):
        H = getH(q,V)
        out = eigen(H.data)
        lam = out[0]
        Q = out[1]
        temp = softabs_map(lam,alpha)
        inv_exp_H = torch.mm(torch.mm(Q,torch.diag(1/temp)),torch.t(Q))
        o = 0.5 * torch.dot(p.data,torch.mv(inv_exp_H,p.data))
        temp2 = 0.5 * torch.log((temp)).sum()
        return(o + temp2)
    return(T_givenq)

def pi_wrap(alpha,return_float):
    def inside(x,y,return_float):
        return(H(x,y,alpha,return_float))
    return inside
def H(q,p,alpha,return_float):
    if return_float:
        return(V(q).data[0] + T(q,alpha)(p))
    else:
        return(V(q) + Variable(T(q,alpha)(p)))

def genleapfrog_wrap(alpha,delta,V):
    def inside(q,p,ep,pi):
        return generalized_leapfrog(q,p,ep,alpha,delta,V)
    return(inside)

def p_sharp(q,p):
    lam, Q = eigen(getH(q, V).data)
    p_s = dtaudp(p.data, alp, lam, Q)
    return(p_s)

alp = 1e6
fi_fake = pi_wrap(alp,True)
gleapfrog = genleapfrog_wrap(alp,0.1,V)

store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = GNUTS(q,0.1,fi_fake,gleapfrog,10,p_sharp)
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
#print("store is {}".format(store))
#print(empCov)
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))
#print(fit)