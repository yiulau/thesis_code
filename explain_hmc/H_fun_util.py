import torch
from generate_momentum_util import T_fun_wrap, generate_momentum_wrap
from genleapfrog_ult_util import dtaudq, dphidq,dtaudp
class H_fun:
    def __init__(self,V,metric,Cov=None,var_vec=None,alpha=None,explicit=False):
        self.metric = metric
        self.Cov = Cov
        self.var_vec = var_vec
        self.alpha = alpha
        self.V = V
        self.T = T_fun_wrap(self.metric,self.Cov,self.var_vec,self.V,self.alpha,False)
        self.generate_momentum = generate_momentum_wrap(self.metric,self.var_vec,self.Cov,self.V,self.alpha,self.q)
        self.dH_dq = self.set_dH_dq()
        self.dH_dp = self.set_dH_dp()
        self.V_float_fun = lambda q: V(q,True)
        self.T_float_fun = T_fun_wrap(self.metric,self.Cov,self.var_vec,self.V,self.alpha,True)
        self.H_float_fun = self.set_H_float_fun()
        self.dG_dt = self.set_dG_dt()
        self.p_sharp_fun = self.set_p_sharp()
        # momentum_cov always a matrix
        self.dG_dt = self.set_dG_dt(self.metric)



    def set_dG_dt(self):
        if(self.metric=="softabs"):
            def dG_dt(p,q,dH,lam,Q,dV):
                return(-torch.dot(q.data,dtaudq(p,dH,Q,lam,self.alpha))+dphidq(lam,self.alpha,dH,Q,dV))
        else:
            def dG_dt(q,p):
                return(2 * self.T_float_fun(p) - torch.dot(q.data,self.dH_dq(q)))

    def set_p_sharp(self):
        if (self.metric == "softabs"):
            def p_sharp(p):
                return(dtaudp(p))
        else:
            def p_sharp(p):
                return(self.dH_dp(p))
    def set_dH_dq(self):
        def dH_dq(q):
            if not q.grad is None:
                q.grad.data.zero_()
            potential = self.V(q)
            potential.backward()
            return(q.grad.data)
        return(dH_dq)

    def set_dH_dp(self):
        if self.metric=="unit_e":
            def dH_dp(p):
                return(p.data)
        elif self.metric=="diag_e":
            def dH_dp(p):
                return(p.data * self.var_vec)
        elif self.metric=="dense_e":
            def dH_dp(p):
                return(torch.mv(self.Cov,p.data))
        elif self.metric=="softabs":
            def dH_dp(p):
                print("softabs should not use this function")
        else:
            return("error")

    def set_H_float_fun(self):
        def H_float(q,p):
            return(self.T_float_fun(q,p)+self.V_float_fun(q))
