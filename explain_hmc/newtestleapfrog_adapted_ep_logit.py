import pandas as pd
import torch, math
from torch.autograd import Variable,grad
import pystan
import numpy
import pickle
import pandas as pd
from leapfrog_ult_util import leapfrog_ult as leapfrog
from leapfrog_ult_util import HMC_alt_ult
from general_util import logsumexp_torch
from adapt_util import dual_averaging_ep
from generate_momentum_util import generate_momentum_wrap
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
generate_momentum = generate_momentum_wrap(metric="unit_e")

store_ep = dual_averaging_ep(tune_l=2000,time=1.4,gamma=0.05,t_0=10,kappa=0.75,
                             target_delta=0.65,sampler_onestep=HMC_alt_ult,
                             generate_momentum=generate_momentum,H_fun=H,
                             integrator=leapfrog,q=q)
store_ep = store_ep/store_ep[len(store_ep)-1]
#import matplotlib.pyplot as plt
#plt.plot(store_ep[-1500:])
#plt.show()
exit()

#ep = 0.1
#num_step = 10

store = torch.zeros((chain_l,dim))
for i in range(chain_l):
    print("round {}".format(i))
    #out = HMC(0.1,10,q)
    out = HMC_alt(ep,num_step,q,leapfrog,pi)
    store[i,] = out[0].data
    #out = NUTS(q,0.1,pi,leapfrog,NUTS_criterion)
    #print("tree depth is {}".format(out[1]))
    #store[i,] = out[0].data
    #store[i,]=out.data
    q.data = out[0].data
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
print("final ep is {}, numstep is {}".format(ep,num_step))
print(fit)