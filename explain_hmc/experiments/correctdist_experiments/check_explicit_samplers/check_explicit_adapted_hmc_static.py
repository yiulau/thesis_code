import torch
from torch.autograd import Variable
import pystan
import numpy
import pickle
import pandas as pd
from explicit.leapfrog_ult_util import leapfrog_ult as leapfrog
from explicit.leapfrog_ult_util import HMC_alt_ult
from explicit.general_util import logsumexp_torch
from explicit.adapt_util import full_adapt
from explicit.generate_momentum_util import generate_momentum_wrap
from experiments.correctdist_experiments.prototype import check_mean_var

dim = 4
num_ob = 25
chain_l = 2000
burn_in = 1000

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
#fit = mod.sampling(data=data,refresh=0)
#print(fit)
#exit()


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







######################
# dual averaging
tune_l = 2000
time = 1.4
gamma = 0.05
t_0 = 10
kappa = 0.75
target_delta = 0.65
metrcs = "dense_e"
covar = torch.eye(dim,dim)
generate_momentum = generate_momentum_wrap(metric=metrcs,Cov=covar)
store_ep,start_q = full_adapt(metric=metrcs,tune_l=250,time=1.4,gamma=0.05,t_0=10,kappa=0.75,
                             target_delta=0.65,sampler_onestep=HMC_alt_ult,
                             generate_momentum=generate_momentum,H_fun=H,V=V,
                             integrator=leapfrog,q=q)
#store_ep,start_q = dual_averaging_ep(tune_l=20,time=1.4,gamma=0.05,t_0=10,kappa=0.75,
#                             target_delta=0.65,sampler_onestep=HMC_alt_ult,
 #                            generate_momentum=generate_momentum,H_fun=H,
  #                           integrator=leapfrog,q=q)

#store_ep = store_ep/store_ep[len(store_ep)-1]
#import matplotlib.pyplot as plt
#plt.plot(store_ep)
#plt.show()
#exit()

# convert to float because store_ep is numpy array
ep = float(store_ep[len(store_ep)-1])
#print(ep)
#exit()
num_step = max(1,round(time/ep))

store = torch.zeros((chain_l,dim))
for i in range(chain_l):
    print("round {}".format(i))
    out = HMC_alt_ult(ep,num_step,q,leapfrog,H,generate_momentum)
    store[i,] = out[0].data
    q.data = out[0].data


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