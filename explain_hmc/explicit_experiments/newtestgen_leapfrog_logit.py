import torch
from torch.autograd import Variable
import numpy as np
import pystan
import pickle
import time
import pandas as pd
#from genleapfrog_ult_util import getH, getdH, getdV, eigen, softabs_map, dphidq, dtaudp, dtaudq, generate_momentum
from explicit.genleapfrog_ult_util import rmhmc_step, getH, eigen, softabs_map
chain_l = 500
burn_in = 100
alp =1e6
dim = 8
num_ob = 532
recompile = False
if recompile:
    mod = pystan.StanModel(file="./alt_log_reg.stan")
    with open('model.pkl', 'wb') as f:
        pickle.dump(mod, f)

mod = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv("./pima_india.csv",header=0,sep=" ")
#print(df)
dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
y_np = dfm[:,8]
y_np = y_np.astype(np.int64)
X_np = dfm[:,1:8]
dim = X_np.shape[1]
num_ob = X_np.shape[0]
#print(y_np)
#print(X_np.shape)
#exit()
#dim =3
#num_ob = 10
#y_np= np.random.binomial(n=1,p=0.5,size=num_ob)
#X_np = np.random.randn(num_ob,dim)
#dim = X_np.shape[1]
#num_ob = X_np.shape[0]
#print(y_np.dtype)

data = dict(y=y_np,X=X_np,N=num_ob,p=dim)
#print(data)

#fit = mod.sampling(data=data,refresh=0)
#print(fit)
#exit()
y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))

def V(q):
    beta = q
    likelihood = torch.dot(beta,torch.mv(torch.t(X),y)) - \
    torch.sum(torch.log(1+torch.exp(torch.mv(X,beta))))
    prior = -torch.dot(beta,beta) * 0.5
    posterior = prior + likelihood
    return(-posterior)


def T(q,alpha):
    def T_givenq(p):
        _,H_ = getH(q,V)
        out = eigen(H_.data)
        lam = out[0]
        Q = out[1]
        temp = softabs_map(lam,alpha)
        inv_exp_H = torch.mm(torch.mm(Q,torch.diag(1/temp)),torch.t(Q))
        o = 0.5 * torch.dot(p.data,torch.mv(inv_exp_H,p.data))
        temp2 = 0.5 * torch.log((temp)).sum()
        return(o + temp2)
    return(T_givenq)

def H(q,p,alpha):
    # returns float
    return(V(q).data[0] + T(q,alpha)(p))



store = torch.zeros((chain_l,dim))

g,H_ = getH(q,V)
lam,Q = eigen(H_.data)

begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = rmhmc_step(q,H,0.1,10,alp,0.1,V)
    store[i,]=out[0].data
    q.data = out[0].data
totalt = time.time() - begin

store = store[burn_in:,]
store = store.numpy()
empCov = np.cov(store,rowvar=False)
emmean = np.mean(store,axis=0)
print("length of chain is {}".format(chain_l))
print("burn in is {}".format(burn_in))
print("total time is {}".format(totalt))
print("alpha is {}".format(alp))
#print(empCov)
#print("store is {}".format(store))
print("sd is {}".format(np.sqrt(np.diagonal(empCov))))
print("mean is {}".format(emmean))

print(fit)