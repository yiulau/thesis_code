import torch
from explicit.genleapfrog_ult_util import eigen, getH, softabs_map
def generate_momentum_wrap(metric,var_vec=None,Cov=None,V=None,alpha=None):
    # returns tensor
    # generate from prob(p given q)
    if (metric=="unit_e"):
        def generate(q):
            return(torch.randn(len(q)))
    elif (metric=="diag_e"):
        sd = torch.sqrt(var_vec)
        inv_sd = 1/sd
        def generate(q):
            return(torch.randn(len(q)) * inv_sd)
    elif (metric =="dense_e"):
        L = torch.potrf(Cov,upper=False)
        L_inv = torch.inverse(L)
        def generate(q):
            return(torch.mv(L_inv,torch.randn(len(q))))
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


def T_fun_wrap(metric,Cov=None,var_vec=None,V=None,alpha=None,returns_float=None):
    if returns_float==False:
        if metric=="unit_e":
            def T(p):
                return(torch.dot(p,p)*0.5)
        elif metric=="dense_e":
            def T(p):
                return(torch.dot(p,torch.mv(Cov,p))*0.5)
        elif metric=="diag_e":
            def T(p):
                return(torch.dot(p,var_vec*p)*0.5)
        elif metric=="softabs":
            def T(p):
                return ("error")
    else:
        if metric=="unit_e":
            def T(p):
                return((torch.dot(p,p)*0.5).data[0])
        elif metric=="dense_e":
            def T(p):
                return((torch.dot(p,torch.mv(Cov,p))*0.5).data[0])
        elif metric=="diag_e":
            def T(p):
                return((torch.dot(p,var_vec*p)*0.5).data[0])
        elif metric=="softabs":
            def T(q,p):
                H = getH(q, V)
                out = eigen(H.data)
                lam = out[0]
                Q = out[1]
                temp = softabs_map(lam, alpha)
                inv_exp_H = torch.mm(torch.mm(Q, torch.diag(1 / temp)), torch.t(Q))
                o = 0.5 * torch.dot(p.data, torch.mv(inv_exp_H, p.data)).data[0]
                temp2 = 0.5 * torch.log((temp)).sum()
                return (o + temp2)
    return(T)

def H_fun_wrap(V,T):
    # this implementation is problematic when metric is softabs. leave this for now
    def H_fun(q,p,return_float):
        if return_float:
            return((V(q)+T(p)).data[0])
        else:
            return(V(q)+T(p))