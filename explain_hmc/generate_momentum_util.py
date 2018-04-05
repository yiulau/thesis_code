import torch
from genleapfrog_ult_util import eigen, getH, softabs_map
def generate_momentum_wrap(metric,var_vec=None,Cov=None,V=None,alpha=None):
    # returns tensor
    if (metric=="unit_e"):
        def generate(q):
            return(torch.randn(len(q)))
    elif (metric=="diag_e"):
        sd = torch.sqrt(var_vec)
        def generate(q):
            return(torch.randn(len(q)) * sd)
    elif (metric =="dense_e"):
        L = torch.potrf(Cov,upper=False)
        def generate(q):
            return(torch.mv(L,torch.randn(len(q))))
    elif (metric =="softabs"):
        def generate(q):
            lam, Q = eigen(getH(q, V).data)
            temp = torch.mm(Q, torch.diag(torch.sqrt(softabs_map(lam, alpha))))
            out = torch.mv(temp, torch.randn(len(lam)))
            return(out)
    else:
        # should raise error here
        return("error")
    return generate
