import pandas as pd
import torch
from torch.autograd import Variable,grad
import pystan
import numpy
import pickle
import time, cProfile, math
dim = 5
num_ob = 100
chain_l = 200
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
#df = pd.read_csv("./pima_india.csv",header=0,sep=" ")
#print(df)
#dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
#y_np = dfm[:,8]
#y_np = y_np.astype(numpy.int64)
#X_np = dfm[:,1:8]
#dim = X_np.shape[1]
num_ob = X_np.shape[0]
data = dict(y=y_np,X=X_np,N=num_ob,p=dim)
fit = mod.sampling(data=data,refresh=0)

#print(fit)

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))
def generate_momentum(alpha,lam,Q):
    # generate by multiplying st normal by QV^(0.5) where Sig = QVQ^T
    #print(lam,Q)
    temp = torch.mm(Q,torch.diag(torch.sqrt(softabs_map(lam,alpha))))
    #print(temp)
    out = torch.mv(temp,torch.randn(len(lam)))
    return(out)
def softabs_map(lam,alpha):
    # takes vector as input
    # returns a vector
    return(lam * coth_torch(lam*alpha))

def coth(x):
    return(1/numpy.asscalar(numpy.tanh(x)))
def coth_torch(x):
    return(1/torch.tanh(x))

def eigen(H):
    # input must be of type tensor ** not variable
    try:
        out = torch.symeig(H,True)
    except RuntimeError:
        #print(fit)
        print(H)
        print(numpy.linalg.eig(H.numpy()))
    return(out[0],out[1])

def getdV(q,V):
    potentialE = V(q)
    g = grad(potentialE, q, create_graph=True)[0]
    return(g)

def getdV_explicit(q,V):
    beta = q
    pihat = torch.sigmoid(torch.mv(X,beta))
    out = X.t().mv(y-pihat)
    return(out)


def getH(q,V):
    g = getdV(q,V)
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        H = Variable(torch.zeros(dim, dim)).cuda()
    else:
        H = Variable(torch.zeros(dim, dim))
    for i in range(dim):
        H[i, :] = grad(g[i], q, create_graph=True)[0]
    return(H)

def getdH(q,V):
    H = getH(q,V)
    dim = len(q)
    if q.data.type() == "torch.cuda.FloatTensor":
        dH = torch.zeros(dim,dim,dim).cuda()
    else:
        dH = torch.zeros(dim,dim,dim)
    for i in range(dim):
        for j in range(dim):
            dH[:, i, j] = grad(H[i, j], q, create_graph=False,retain_graph=True)[0].data

    return(dH)

def dphidq(lam,alpha,dH,Q,dV):
    N = len(lam)
    #print("lam is {}".format(lam))
    Jm = J(lam,alpha,len(lam))
    R = torch.diag(1/(lam*coth_torch(alpha*lam)))
    M = torch.mm(Q,torch.mm(R*Jm,torch.t(Q)))
    #print("M is {}".format(M))
    #print("dH is {}".format(dH[0,:,:]))
    #print("trace(MdH) is {}".format(torch.trace(torch.mm(M,dH[0,:,:])) ))
    #print("dV is {}".format(dV))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(torch.mm(M,dH[i,:,:])) + dV[i]
    return(delta)

def J(lam,alpha,length):
    J = torch.zeros(length,length)
    #mindif = 1
    for i in range(length):
        for j in range(length):
            if i!=j:
                #dif = abs(lam[i]-lam[j])
                #if dif < mindif:
                    #mindif = dif
                J[i,j] = (lam[i]*coth(alpha*lam[i]) - lam[j]*coth(alpha*lam[j]))/(lam[i]-lam[j])
            else:
                J[i,j] = (coth(alpha*lam[i]) + lam[i]*(1-numpy.square(coth(alpha*lam[i])))*alpha)
    #print("mindif is {}".format(mindif))
    return(J)

def D(p,Q,lam,alpha):
    return(torch.diag(torch.mv(torch.t(Q),p)/(lam*coth_torch(alpha*lam))))
def dtaudq(p,dH,Q,lam,alpha):
    N = len(p)
    Jm = J(lam,alpha,len(p))
    Dm = D(p,Q,lam,alpha)
    M = torch.mm(Q,torch.mm(Dm,torch.mm(Jm,torch.mm(Dm,torch.t(Q)))))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(-torch.mm(M,dH[i,:,:]))

    return(delta)

def dtaudp(p,alpha,lam,Q):
    return(Q.mv(torch.diag(1/softabs_map(lam,alpha)).mv((torch.t(Q).mv(p)))))

def V(q):
    beta = q
    likelihood = torch.dot(beta,torch.mv(torch.t(X),y)) - \
    torch.sum(torch.log(1+torch.exp(torch.mv(X,beta))))
    prior = -torch.dot(beta,beta) * 0.5
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
        #print("o is {}".format(o))

        return(o + temp2)
    return(T_givenq)

def pi_wrap(alpha):
    def inside(x,y):
        return(H(x,y,alpha))
    return inside
def H(q,p,alpha):
    return(V(q).data[0] + T(q,alpha)(p))

def genleapfrog_wrap(alpha,delta,V):
    def inside(q,p,ep,pi):
        return generalized_leapfrog(q,p,ep,alpha,delta,V)
    return(inside)
def generalized_leapfrog(q,p,epsilon,alpha,delta,V):

    lam,Q = eigen(getH(q,V).data)
    dH = getdH(q,V)
    dV = getdV(q,V)

    p.data = p.data - epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV.data)
    #print("dphidq is {}".format( dphidq(lam,alpha,dH,Q,dV.data)))
    #return (q, p)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        pprime = rho - epsilon * 0.5 * dtaudq(p.data,dH,Q,lam,alpha)
        deltap = torch.max(torch.abs(p.data-pprime))
        p.data = pprime.clone()

        count = count + 1
        #print("p is {}".format(p.data))

    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5
    olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):

        lam,Q = eigen(getH(q,V).data)
        qprime = sigma.data + 0.5 * epsilon * dtaudp(p.data,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p.data,alpha,lam,Q)
        deltaq = torch.max(torch.abs(q.data-qprime))
        q.data = qprime.clone()
        count = count + 1

    dH = getdH(q,V)
    dV = getdV(q,V)
    lam,Q = eigen(getH(q,V).data)
    p.data = p.data - 0.5 * dtaudq(p.data,dH,Q,lam,alpha) * epsilon
    p.data = p.data - 0.5 * dphidq(lam,alpha,dH,Q,dV.data) * epsilon

    #print("p is {}".format(p.data))
    return(q,p)
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
    #log_w = -pi(q_init,p).data.numpy()
    log_w = -pi(q_init,p)
    sum_p = p.clone()
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, s_prime, log_w_prime,sum_dp = BuildTree(q_left, p_left, -1, j, epsilon, leapfrog, pi,
                                                                            NUTS_criterion)
        else:
            _, _, q_right, p_right, q_prime, s_prime, log_w_prime, sum_dp = BuildTree(q_right, p_right, 1, j, epsilon, leapfrog, pi,
                                                                              NUTS_criterion)
        if s_prime:
            accep_rate = numpy.exp(min(0,(log_w_prime-log_w)))
            u = numpy.random.rand(1)
            if u < accep_rate:
                q_prop.data = q_prime.data.clone()
        log_w = logsumexp(log_w,log_w_prime)
        sum_p = sum_p + sum_dp
        #s = s_prime and NUTS_criterion(q_left,q_right,p_left,p_right)
        lam,Q = eigen(getH(q_left,V).data)
        p_sleft = dtaudp(p_left.data,alp,lam,Q)
        lam,Q = eigen(getH(q_right,V).data)
        p_sright = dtaudp(p_right.data,alp,lam,Q)
        s = s_prime and gen_NUTS_criterion(p_sleft,p_sright,sum_p)
        j = j + 1
        s = s and (j<max_tdepth)

    return(q_prop,j)

def BuildTree(q,p,v,j,epsilon,leapfrog,pi,NUTS_criterion):
    if j ==0:
        q_prime,p_prime = leapfrog(q,p,v*epsilon,pi)
        #log_w_prime = -pi(q_prime.data,p_prime.data)
        #log_w_prime = -pi(q_prime, p_prime).data.numpy()
        log_w_prime = -pi(q_prime,p_prime)
        return q_prime, p_prime, q_prime, p_prime, q_prime, True, log_w_prime,p_prime
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime,sum_p = BuildTree(q, p, v, j - 1, epsilon, leapfrog, pi, NUTS_criterion)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,s_dprime,log_w_dprime,sum_dp = BuildTree(q_left,p_left,v,j-1,epsilon,leapfrog,pi,NUTS_criterion)
            else:
                _, _, q_right, p_right, q_dprime, s_dprime, log_w_dprime,sum_dp = BuildTree(q_right, p_right, v, j - 1, epsilon,
                                                                                 leapfrog, pi, NUTS_criterion)
            accep_rate = numpy.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
            u = numpy.random.rand(1)[0]
            if u < accep_rate:
                q_prime.data = q_dprime.data.clone()
            #s_prime = s_dprime and NUTS_criterion(q_left,q_right,p_left,p_right)
            sum_p = sum_p + sum_dp
            lam, Q = eigen(getH(q_left,V).data)
            p_sleft = dtaudp(p_left.data, alp, lam, Q)
            lam, Q = eigen(getH(q_right,V).data)
            p_sright = dtaudp(p_right.data, alp, lam, Q)
            s_prime = s_dprime and gen_NUTS_criterion(p_sleft, p_sright, sum_p)
            log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, s_prime, log_w_prime,sum_p

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

def gen_NUTS_criterion(p_left,p_right,p_sum):
    o = (torch.dot(p_left,p_sum.data) > 0) and \
        (torch.dot(p_right,p_sum.data) > 0)
    return(o)

def NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue going
    # False = stops
    o = (torch.dot(q_right.data-q_left.data,p_right.data) >=0) or \
        (torch.dot(q_right.data-q_left.data,p_left.data) >=0)
    return(o)

#q = Variable(torch.randn(dim),requires_grad=True)

v = -1
#q_clone = q.clone()
#epsilon = 0.11
alp = 1e6
#print("q is {}".format(q))
fi_fake = pi_wrap(alp)
gleapfrog = genleapfrog_wrap(alp,0.1,V)
#for _ in range(4):
#    out = gleapfrog(q, p, 0.1, fi_fake)
#    q.data = out[0].data
#    p.data = out[1].data
#o = gleapfrog(q,p,0.1,fi_fake)
#print("10 gleapforg q {}".format(q))
#o = NUTS(q_clone,0.1,fi_fake,gleapfrog,NUTS_criterion)
#print("propsed q {}".format(o))
#exit()
#print(o)
store = torch.zeros((chain_l,dim))
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    #out = NUTS(q,0.12,pi,leapfrog,NUTS_criterion)
    out = NUTS(q,0.1,fi_fake,gleapfrog,NUTS_criterion)
    store[i,] = out[0].data # turn this on when using Nuts
    q.data = out[0].data # turn this on when using nuts
    print("q is {} tree length {}".format(q.data,out[1]))
total = time.time() - begin
print("total time is {}".format(total))
print("length of chain is {}".format(chain_l))
print("length of burn in is {}".format(burn_in))
print("Use logit")
store = store[burn_in:,]
store = store.numpy()
empCov = numpy.cov(store,rowvar=False)
emmean = numpy.mean(store,axis=0)
print("store is {}".format(store))
#print(empCov)
print("sd is {}".format(numpy.sqrt(numpy.diagonal(empCov))))
print("mean is {}".format(emmean))
print(fit)