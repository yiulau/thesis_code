import torch
import numpy as np

lower_softabs_thresh = 0.001
upper_softabs_thresh = 10000
def coth_torch(x):
    return(1/torch.tanh(x))

def coth(x):
    return(1/np.asscalar(np.tanh(x)))

def softabs_map(lam,alpha):
    # takes vector as input
    # returns a vector
    return(lam * coth_torch(lam*alpha))

def softabs_map_stable(lam,alpha):
    out = torch.zeros(len(lam))
    for i in range(len(lam)):
        alp_lam = lam[i] * alpha
        if (abs(alp_lam)<lower_softabs_thresh):
            out[i] = (1. + (1./3.) * alp_lam * alp_lam)/alpha
        elif (abs(alp_lam)>upper_softabs_thresh):
            out[i] = abs(lam[i])
        else:
            out[i] = lam[i] * coth(alp_lam)
    return(out)

lam = torch.randn(4)

alpha = 1e6

o1 = softabs_map(lam,alpha)
print(o1)
o2 = softabs_map_stable(lam,alpha)
print(o2)