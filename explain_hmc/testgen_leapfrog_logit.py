import torch
from torch.autograd import Variable,grad
import numpy as np
import pystan
import pickle
import time,cProfile


chain_l = 200
burn_in = 100
alp =1e6
dim = 4
num_ob = 25
recompile = False
if recompile:
    mod = pystan.StanModel(file="./alt_log_reg.stan")
    with open('model.pkl', 'wb') as f:
        pickle.dump(mod, f)

mod = pickle.load(open('model.pkl', 'rb'))


y_np= np.random.binomial(n=1,p=0.5,size=num_ob)
X_np = np.random.randn(num_ob,dim)

data = dict(y=y_np,X=X_np,N=num_ob,p=dim)
fit = mod.sampling(data=data,refresh=0)
print(fit)

y = Variable(torch.from_numpy(y_np).float(),requires_grad=False)

X = Variable(torch.from_numpy(X_np).float(),requires_grad=False)

q = Variable(torch.randn(dim),requires_grad=True)
p = Variable(torch.randn(dim))
def generate_momentum(alpha,lam,Q):
    # generate by multiplying st normal by QV^(0.5) where Sig = QVQ^T
    temp = torch.mm(Q,torch.diag(1./torch.sqrt(softabs_map(lam,alpha))))
    out = torch.mv(temp,torch.randn(len(lam)))
    return(out)
def softabs_map(lam,alpha):
    # takes vector as input
    # returns a vector
    return(1/torch.tanh(lam*alpha))

def coth(x):
    return(1/np.asscalar(np.tanh(x)))
def coth_torch(x):
    return(1/torch.tanh(x))

def eigen(H):
    # input must be of type tensor ** not variable
    #out = torch.eig(H,True)
    out = torch.symeig(H,True)
    return(out[0][:,0],out[1])

def getdV(q,V):
    potentialE = V(q)
    g = grad(potentialE, q, create_graph=True)[0]
    return(g)

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
    #dH = Variable(torch.zeros(dim, dim, dim))
    if q.data.type() == "torch.cuda.FloatTensor":
        dH = torch.zeros(dim,dim,dim).cuda()
    else:
        dH = torch.zeros(dim,dim,dim)
    #count = 0
    for i in range(dim):
        for j in range(dim):
            #print(i,j)
            dH[:, i, j] = grad(H[i, j], q, create_graph=False,retain_graph=True)[0].data
            #try:
            #    dH[:,i, j] = grad(H[i, j], q, create_graph=False,retain_graph=True)[0].data
            #except RuntimeError:
            #    dH[:,i, j] = torch.zeros(len(q))

    return(dH)

def dphidq(lam,alpha,dH,Q,dV):
    N = len(lam)
    Jm = J(lam,alpha,len(lam))
    R = torch.diag(1/(lam*coth_torch(alpha*lam)))
    M = torch.mm(Q,torch.mm(R*Jm,torch.t(Q)))
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
                J[i,j] = (lam[i]*coth(alpha*lam[i]) - lam[j]*coth(alpha*lam[j]))
            else:
                J[i,j] = (coth(alpha*lam[i]) - lam[i]*(1-np.square(coth(alpha*lam[i])))*alpha)
    #print("mindif is {}".format(mindif))
    return(J)

def D(p,Q,lam,alpha):
    #return(torch.diag(torch.mv(torch.t(Q),p)/(lam*coth_torch(alpha*lam))))
    return (torch.diag(torch.mv(torch.t(Q), p)))
def dtaudq(p,dH,Q,lam,alpha):
    N = len(p)
    Jm = J(lam,alpha,len(p))
    #print("eigenvalues {}".format(lam))
    #print("J {}".format(Jm))
    Dm = D(p,Q,lam,alpha)
    #print("D {}".format(Dm))
    M = torch.mm(Q,torch.mm(Dm,torch.mm(Jm,torch.mm(Dm,torch.t(Q)))))
    #print("M is {}".format(M))
    delta = torch.zeros(N)
    for i in range(N):
        delta[i] = 0.5 * torch.trace(-torch.mm(M,dH[i,:,:]))

    return(delta)

def dtaudp(p,alpha,lam,Q):
    return(torch.mv(Q,(softabs_map(lam,alpha)*torch.mv(torch.t(Q),p))))

def V(q):
    beta = q

    pihat = torch.sigmoid(torch.mv(X,beta))

    likelihood = torch.dot(torch.log(pihat),y) + torch.dot((1-y),torch.log(1-pihat))

    prior = - torch.dot(beta,beta) * 0.5

    posterior = prior + likelihood

    return(-posterior)

def T(q,alpha):
    def T_givenq(p):
        H = getH(q,V)
        out = eigen(H.data)
        lam = out[0]
        Q = out[1]
        temp = softabs_map(lam,alpha)
        inv_exp_H = torch.mm(torch.mm(Q,torch.diag(temp)),torch.t(Q))
        o = 0.5 * torch.dot(p.data,torch.mv(inv_exp_H,p.data))
        temp2 = 0.5 * torch.log(torch.abs(temp)).sum()
        return(o + temp2)
    return(T_givenq)

def H(q,p,alpha):
    return(V(q).data[0] + T(q,alpha)(p))

def generalized_leapfrog(q,p,epsilon,alpha,delta,V):
    q = q.clone()
    p = p.clone()
    #print("break1")
    lam,Q = eigen(getH(q,V).data)
    #print("break2")
    dH = getdH(q,V)
    #print("break3")
    dV = getdV(q,V)
    #print("break4")
    #print(dphidq(lam,alpha,dH,Q,dV.data))
    #print(dH,dV)
    #print(q,p)
    #tempout = dphidq(lam,alpha,dH.data,Q,dV.data)
    #print("should be {},get {}".format(q.data,tempout))
    p.data = p.data - epsilon * 0.5 * dphidq(lam,alpha,dH,Q,dV.data)
    #print("break5")
    #p.data = p.data - epsilon * 0.5 * tempout
    #print(q,p)
    rho = p.data.clone()
    pprime = p.data.clone()
    deltap = delta + 0.5
    #print(q,p)
    #print(dH)
    count = 0

    while (deltap > delta) and (count < 5):
        pprime = rho - epsilon * 0.5 * dtaudq(p.data,dH,Q,lam,alpha)
        #print(dtaudq(p.data,dH,Q,lam,alpha))
        #tempout =  dtaudq(p.data,dH.data,Q,lam,alpha)
        #print("should be {},get {}".format(q.data,tempout))
        deltap = torch.max(torch.abs(p.data-pprime))
        #print(deltap)
        p.data = pprime.clone()
        count = count + 1
        #print(p)
    #print(q,p)
    print("break6")
    sigma = Variable(q.data.clone(),requires_grad=True)
    qprime = q.data.clone()
    deltaq = delta + 0.5
    olam,oQ = eigen(getH(sigma,V).data)
    count = 0
    while (deltaq > delta) and (count < 5):

        lam,Q = eigen(getH(q,V).data)
        print("break6b")
        qprime = sigma.data + 0.5 * epsilon * dtaudp(p.data,alpha,olam,oQ) + 0.5 * epsilon* dtaudp(p.data,alpha,lam,Q)
        print("break6c")
        deltaq = torch.max(torch.abs(q.data-qprime))
        print("break6d")
        print("deltaq is {}".format(deltaq))
        print("deltaq > delta is {}".format(deltaq > delta))
        q.data = qprime.clone()
        count = count + 1
        print("count is {} ".format(count))
    #print(q,p)
    print("break7")
    dH = getdH(q,V)
    print("break8")
    dV = getdV(q,V)
    print("break9")
    #print("got here")
    #exit()
    lam,Q = eigen(getH(q,V).data)
    print("break10")
    p.data = p.data - 0.5 * dtaudq(p.data,dH,Q,lam,alpha) * epsilon
    print("break11")
    p.data = p.data - 0.5 * dphidq(lam,alpha,dH,Q,dV.data) * epsilon
    print("break12")
    #print(q,p)
    return(q,p)

def rmhmc_step(initq,H,epsilon,L,alpha,delta,V):
    #p = Variable(torch.randn(len(initq)),requires_grad=True)
    q = Variable(initq.data.clone(), requires_grad=True)
    lam,Q = eigen(getH(q,V).data)
    p = Variable(generate_momentum(alpha,lam,Q))
    current_H = (V(q).data + T(q,alpha)(p)).numpy()[0]
    #print("current H {}".format(current_H))

    for _ in range(L):
        out = generalized_leapfrog(q,p,epsilon,alpha,delta,V)
        q.data = out[0].data
    #proposed_H = out[2]
    proposed_H = (V(q).data + T(initq,alpha)(out[1])).numpy()[0]
    #print("proposed H {}".format(proposed_H))
    #exit()
    u = np.random.rand(1)
    print("current H {}".format(current_H))
    print("propsed H {}".format(proposed_H))
    print("accep rate {}".format(np.exp(current_H-proposed_H)))
    if u < np.exp(current_H - proposed_H):
        return(out[0])
    else:
        return(q)

store = torch.zeros((chain_l,dim))
a = getH(q,V).data.numpy()
print(a)
print(np.allclose(a, a.T, atol=1e-8))
print(torch.eig(getH(q,V).data),True)
print(torch.symeig(getH(q,V).data),True)
print(np.linalg.eig(getH(q,V).data))
#cProfile.run("rmhmc_step(q,H,0.1,10,alp,0.1,V)")
exit()
begin = time.time()
for i in range(chain_l):
    print("round {}".format(i))
    #out = HMC(0.1,10,q)
    #out = HMC_alt(0.1,10,q,leapfrog,pi)
    #out = NUTS(q,0.1,pi,leapfrog,NUTS_criterion)
    out = rmhmc_step(q,H,0.1,10,alp,0.1,V)
    #print("tree depth is {}".format(out[1]))
    #store[i,] = out[0].data
    store[i,]=out.data
    q.data = out.data
    #q.data = out[0].data
    #print("q is {}".format(q.data))
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
print("sd is {}".format(np.sqrt(np.diagonal(empCov))))
print("mean is {}".format(emmean))

print(fit)