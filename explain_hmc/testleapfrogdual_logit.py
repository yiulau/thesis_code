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
#fit = mod.sampling(data=data,refresh=0)
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
def norm_logsumexp(a,b):
    s = max(a,b)
    out = s + numpy.log(numpy.exp(a-s) + numpy.exp(b-s))
    return(out)
##exit()
def pi(q,p):
    beta = q
    #print("xbeta {}".format((torch.mv(X, beta))))
    term1 = torch.sum(torch.log(1 + torch.exp(torch.mv(X, beta))))
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
    current_H = pi(q, p)
    for _ in range(L):
        temp_q, temp_p = leapfrog(q, p, epsilon, pi)
        q.data, p.data = temp_q.data.clone(), temp_p.data.clone()

    proposed_H = pi(q, p)
    temp = (current_H - proposed_H)

    #print("current H is {}".format(current_H))
    #print("proposed H is {}".format(proposed_H))
    accep_rate =  numpy.exp(min(0,temp.data.numpy()))
    #print("accept rate  is {}".format((numpy.asscalar(accep_rate))))
    if (numpy.log(numpy.random.random(1)) < numpy.asscalar(accep_rate)):
        return (q,accep_rate)
    else:
        return (current_q,accep_rate)

def find_reasonable_ep(q):
    ep = 1
    p = Variable(torch.randn(len(q)))
    qprime,pprime = leapfrog(q,p,ep,pi)
    a = 2 * (-pi(qprime,pprime).data.numpy() + pi(q,p).data.numpy() > numpy.log(0.5)) - 1
    a = a[0]
    while a * (-pi(qprime, pprime).data.numpy() + pi(q, p).data.numpy()) > (-a * numpy.log(2)):
        ep = numpy.exp2(a) * ep
        qprime,pprime = leapfrog(q,p,ep,pi)
    return(ep)



#print(find_reasonable_ep(q))
#exit()
#p = Variable(torch.randn(dim),requires_grad=True)
#out = HMC_alt(0.1,10,q,leapfrog,pi)
#print(out)
#exit()
#print("q is {}".format(q.data))
#print("torch output {}".format(torch.dot(q.data,q.data)))
#print(numpy.dot(q.data.numpy(),q.data.numpy()))
#print("p is {}".format(p.data))
#print("torch.output{}".format(torch.dot(p.data,p.data)))
#print(numpy.dot(p.data.numpy(),p.data.numpy()))
#print("H is {}".format(pi(q,p).data.numpy()))
#exit()
#print(pi(q,p))
#exit()
######################
# dual averaging
tune_l = 2000
time = 1.4
ep = find_reasonable_ep(q)
mu = numpy.log(10 * ep)
bar_ep_i = 1
bar_H_i = 0
gamma = 0.05
t_0 = 10
kappa = 0.75
target_delta = 0.65
store_ep = numpy.zeros(tune_l)
for i in range(tune_l):
    num_step = numpy.int(max(1.,numpy.round(time/ep)))
    print("i is {}, num_step is {}, ep is {}".format(i,num_step,ep))
    out = HMC_alt(ep, num_step , q, leapfrog, pi)
    q.data = out[0].data
    alpha = numpy.asscalar(out[1])
    #print("alphatype is {}".format(type(alpha)))
    #print("alpha is {}".format(alpha))
    bar_H_i = (1-1/(i+ 1 +t_0)) * bar_H_i + (1/(i + 1+ t_0)) * (target_delta - alpha)
    #print("bar_H_i {}".format(bar_H_i))
    #exit()
    logep = mu - numpy.sqrt(i+1)/gamma * bar_H_i
    #print("logep is {}".format(logep))
    logbarep = numpy.power(i+1,-kappa) * logep + (1-numpy.power(i+1,-kappa))* numpy.log(bar_ep_i)
    #print("logbarep is {}".format(logbarep))
    bar_ep_i = numpy.exp(logbarep)
    store_ep[i] = bar_ep_i
    #print("bar_ep_i is {}".format(bar_ep_i))
    ep = bar_ep_i

print(ep)
exit()
store_ep = store_ep/ep
import matplotlib.pyplot as plt
plt.plot(store_ep[-1500:])
plt.show()
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