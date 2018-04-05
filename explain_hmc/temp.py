import pandas as pd
import torch,math
from torch.autograd import Variable,grad
from leapfrog_ult_util import HMC_alt_ult, leapfrog_ult
from general_util import logsumexp_torch, logsumexp
import pystan
import numpy
import pickle
import pandas as pd
from generate_momentum_util import T_fun_wrap
import cProfile
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
#fit = mod.sampling(data=data,refresh=0)


y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim),requires_grad=False)
p_ = Variable(p.data.clone(),requires_grad=True)
q_ = Variable(q.data.clone(),requires_grad=True)
def V(beta):
    likelihood = torch.dot(beta, torch.mv(torch.t(X), y)) - \
                 torch.sum(logsumexp_torch(Variable(torch.zeros(num_ob)), torch.mv(X, beta)))
    prior = -torch.dot(beta, beta) * 0.5
    posterior = prior + likelihood
    return(-posterior)

def T(p):
    return(torch.dot(p,p)*0.5)

T = T_fun_wrap(sd=Variable(torch.ones(dim)),metric="diag_e")
def H(q,p,return_float):
    if return_float:
        return((V(q)+T(p)).data[0])
    else:
        return((V(q)+T(p)))
def leapfrog_ult(q,p,epsilon,H_fun):
    # Input:
    # q, p pytorch variables
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)
    H = H_fun(q,p,return_float=False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    q.data += epsilon * p.data
    H = H_fun(q,p,return_float=False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    return(q,p)
def leapfrog_mg(q,p,epsilon,H_fun):
    # Input:
    # q, p pytorch variables
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)
    H = H_fun(q,p,return_float=False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    p.grad.data.zero_()
    H = H_fun(q,p,return_float=False)
    H.backward()
    q.data += epsilon * p.grad.data
    q.grad.data.zero_()
    p.grad.data.zero_()
    H = H_fun(q,p,return_float=False)
    H.backward()
    p.data -= q.grad.data * 0.5 * epsilon
    q.grad.data.zero_()
    return(q,p)

out1 = leapfrog_ult(q,p,0.1,H)

out2 = leapfrog_mg(q_,p_,0.1,H)


exit()
store = torch.zeros((chain_l,dim))
for i in range(chain_l):
    print("round {}".format(i))
    out = HMC_alt_ult(0.1,10,q,leapfrog_ult,H,False)
    store[i,]=out[0]
    q.data = out[0]


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