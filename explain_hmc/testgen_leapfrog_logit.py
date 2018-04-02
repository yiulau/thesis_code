import torch
from torch.autograd import Variable,grad
import numpy as np
import pystan
import pickle
import time,cProfile
import pandas as pd

chain_l = 50
burn_in = 10
alp =1e6
dim = 8
num_ob = 532
recompile = False
if recompile:
    mod = pystan.StanModel(file="./alt_log_reg.stan")
    with open('model.pkl', 'wb') as f:
        pickle.dump(mod, f)

mod = pickle.load(open('model.pkl', 'rb'))

#df = pd.read_csv("./pima_india.csv",header=0,sep=" ")
#print(df)
#dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
#y_np = dfm[:,8]
#y_np = y_np.astype(np.int64)
#X_np = dfm[:,1:8]
#dim = X_np.shape[1]
#num_ob = X_np.shape[0]
#print(y_np)
#print(X_np.shape)
#exit()
dim =3
num_ob = 10
y_np= np.random.binomial(n=1,p=0.5,size=num_ob)
X_np = np.random.randn(num_ob,dim)
dim = X_np.shape[1]
num_ob = X_np.shape[0]
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
    return(1/np.asscalar(np.tanh(x)))
def coth_torch(x):
    return(1/torch.tanh(x))

def eigen(H):
    # input must be of type tensor ** not variable
    try:
        out = torch.symeig(H,True)
    except RuntimeError:
        #print(fit)
        print(H)
        print(np.linalg.eig(H.numpy()))
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

def getH_explicit(q,V):
    beta = q
    pihat = torch.sigmoid(torch.mv(X,beta))
    out = torch.mm(X.t(),torch.mm(X.t(),torch.diag(pihat * (1.- pihat))).t()).data + torch.diag(torch.ones(len(beta)))
    return(out)

def getdH_explicit(q,V):
    beta = q
    dim = len(q)
    dH = torch.zeros(dim,dim,dim)
    pihat = torch.sigmoid(torch.mv(X,beta))
    for i in range(dim):
        dH[i,:,:] = (torch.mm(X.t(),torch.diag(pihat * (1.- pihat))).mm(torch.diag(1.-2*pihat)*X[:,i]).mm(X)).data
    return(dH)
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
                J[i,j] = (coth(alpha*lam[i]) + lam[i]*(1-np.square(coth(alpha*lam[i])))*alpha)
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

def H(q,p,alpha):
    return(V(q).data[0] + T(q,alpha)(p))

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

def rmhmc_step(initq,H,epsilon,L,alpha,delta,V):
    q = Variable(initq.data.clone(), requires_grad=True)
    #print("H {}".format(getH(q, V).data))
    lam,Q = eigen(getH(q,V).data)
    p = Variable(generate_momentum(alpha,lam,Q))
    #print(p)
    current_H = (V(q).data + T(q,alpha)(p)).numpy()[0]
    #print("p is {}".format(p.data))
    #print("q is {}".format(q.data))
    #print(q,p)
    for _ in range(L):
        out = generalized_leapfrog(q,p,epsilon,alpha,delta,V)
        q.data = out[0].data
        p.data = out[1].data
        #print(q,p)
        #print("num step i s {}".format(_))
        #print("p is {}".format(p.data))
    #exit()
    # may need to switch back to T(initq,alpha)
    proposed_H = (V(q).data + T(q,alpha)(p)).numpy()[0]
    #print("pp is {}".format(p.data))
    #print("pq is {}".format(q.data))
    u = np.random.rand(1)
    print("current H {}".format(current_H))
    print("propsed H {}".format(proposed_H))
    print("accep rate {}".format(np.exp(min(0,(current_H-proposed_H)))))
    if np.log(u) < min(0,(current_H - proposed_H)):
        return(q)
    else:
        return(initq)
#out = rmhmc_step(q,H,0.1,10,alp,0.1,V)
#print("auto is {}".format(getdV(q,V)))
#print("explicit is {}".format(getdV_explicit(q,V)))
#print("explicit is {}".format(getH_explicit(q,V)))
#print("auto is {}".format(getH(q,V)))
print(torch.abs(getdH_explicit(q,V) - getdH(q,V)).sum())
exit()
print("explicit is {}".format(getdH_explicit(q,V)))
print("auto is {}".format(getdH(q,V)))
exit()
#lam,Q = eigen(getH(q,V).data)
#p = Variable(generate_momentum(alp,lam,Q))
#print("q is {}".format(q))
#print("p is {}".format(p))
#print((V(q).data + T(q,alp)(p)).numpy()[0])
#out = generalized_leapfrog(q,p,0.01,alp,0.1,V)
#print((V(out[0]).data + T(out[0],alp)(out[1])).numpy()[0])
#print("q is {}".format(out[0]))
#print("p is {}".format(out[1]))
#exit()
store = torch.zeros((chain_l,dim))
a = getH(q,V).data

lam,Q = eigen(a)

begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    out = rmhmc_step(q,H,0.1,10,alp,0.1,V)
    store[i,]=out.data
    q.data = out.data
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
print("store is {}".format(store))
print("sd is {}".format(np.sqrt(np.diagonal(empCov))))
print("mean is {}".format(emmean))

#print(fit)