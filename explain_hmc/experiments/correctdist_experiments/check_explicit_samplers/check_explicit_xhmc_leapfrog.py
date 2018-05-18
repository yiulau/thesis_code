import pandas as pd
import torch
from torch.autograd import Variable
import pystan
import numpy
import pickle
import time
from explicit.general_util import logsumexp_torch
from explicit.leapfrog_ult_util import leapfrog_ult
from explicit.nuts_util import NUTS_xhmc
from experiments.correctdist_experiments.prototype import check_mean_var

dim = 5
num_ob = 100
chain_l = 1000
burn_in = 500
max_tdepth = 10

stan_sampling = False
y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
X_np = numpy.random.randn(num_ob,dim)
address = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data/pima_india.csv"

df = pd.read_csv(address,header=0,sep=" ")
#print(df)
dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
y_np = dfm[:,8]
y_np = y_np.astype(numpy.int64)
X_np = dfm[:,1:8]
dim = X_np.shape[1]
num_ob = X_np.shape[0]

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

def T(p):
    return(torch.dot(p,p)*0.5)


def H(q,p,return_float):
    if return_float:
        return((V(q)+T(p)).data[0])
    else:
        return((V(q)+T(p)))

def dG_dt(q,p):
    # q_grad is dU/dq
    # exact form depends on the momentum distribution ika the metric
    H_fun = H(q, p,False)
    H_fun.backward()
    q_grad = q.grad.data.clone()
    q.grad.data.zero_()
    return(torch.dot(p.data,p.data) - torch.dot(q.data,q_grad))







store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = NUTS_xhmc(q,0.12,H,leapfrog_ult,10,dG_dt,0.1)
    store[i,] = out[0].data # turn this on when using Nuts
    q.data = out[0].data # turn this on when using nuts
    print("q is {} tree length {}".format(q.data, out[1]))
total = time.time() - begin
print("total time is {}".format(total))
print("length of chain is {}".format(chain_l))
print("length of burn in is {}".format(burn_in))
print("Use logit")
store = store[burn_in:,]
store = store.numpy()

mcmc_samples = store
correct = pickle.load(open("result_from_long_chain.pkl", 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()

out = check_mean_var(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)