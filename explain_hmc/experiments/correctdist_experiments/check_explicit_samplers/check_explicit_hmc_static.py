import torch
from torch.autograd import Variable
from explicit.leapfrog_ult_util import HMC_alt_ult, leapfrog_ult
from explicit.general_util import logsumexp_torch
import pystan
import numpy
import pickle
import pandas as pd
from experiments.correctdist_experiments.prototype import check_mean_var
dim = 4
num_ob = 25
chain_l = 500
burn_in = 100


#y_np= numpy.random.binomial(n=1,p=0.5,size=num_ob)
#X_np = numpy.random.randn(num_ob,dim)
address = "/home/yiulau/work/thesis_code/explain_hmc/input_data/pima_india.csv"
#address = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data/pima_india.csv"
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
data = dict(y=y_np,X=X_np,N=num_ob,p=dim)




y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim),requires_grad=False)

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

def generate_momentum(q):
    return(torch.randn(len(q)))
store = torch.zeros((chain_l,dim))
for i in range(chain_l):
    print("round {}".format(i))
    #out = HMC_alt_ult(0.1,10,q,leapfrog_ult,H,False)
    out = HMC_alt_ult(epsilon=0.1,L=10,current_q=q,leapfrog=leapfrog_ult,H_fun=H,generate_momentum=generate_momentum)
    store[i,]=out[0].data
    q.data = out[0].data


store = store[burn_in:,]
store = store.numpy()

mcmc_samples = store
correct = pickle.load(open("../result_from_long_chain.pkl", 'rb'))
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