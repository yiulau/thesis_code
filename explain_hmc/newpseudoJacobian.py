import torch
import numpy as np
def coth(x):
    return(1/np.asscalar(np.tanh(x)))


jacobian_thresh = 0.001
lower_softabs_thresh = 0.0001
upper_softabs_thresh =  1000
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
                J[i,j] = (coth(alpha*lam[i]) + lam[i]*(1-(coth(alpha*lam[i]))**2)*alpha)
    #print("mindif is {}".format(mindif))
    return(J)

def J_2(lam,alpha,length):
    J = torch.zeros(length,length)
    #mindif = 1
    i = 0
    while i < length:
        j =0
        while j <=i:
            #print(i,j)
            dif = abs(lam[i]-lam[j])
            #print(dif)
            if dif < jacobian_thresh:
                alp_lam = lam[i] * alpha
                if alp_lam < lower_softabs_thresh:
                    J[i,j] = (2./3.) * alp_lam * (1.- (2./15.)*alp_lam*alp_lam)
                elif alp_lam > upper_softabs_thresh:
                    # 1 if lam > 0 , -1 otherwise
                    J[i,j] = 2*float(lam[i]>0)-1
                else:
                    J[i, j] = (coth(alpha * lam[i]) + lam[i] * (1 - (coth(alpha * lam[i]))**2) * alpha)
            else:
                J[i,j] = (lam[i]*coth(alpha*lam[i]) - lam[j]*coth(alpha*lam[j]))/(lam[i]-lam[j])
            j = j + 1
        i = i + 1
    #print("mindif is {}".format(mindif))
    return(J)


lam = [1.,2.,3.,4.]
alpha = 1e6
out1 = J(lam,alpha,4)
out2 = J_2(lam,alpha,4)

print(out1)
#print(out2)
print((out2.t() + out2) - torch.diag(torch.diag(out2)))