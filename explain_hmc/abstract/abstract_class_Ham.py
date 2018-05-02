import torch
from abstract.T_unit_e import T_unit_e
from abstract.T_dense_e import T_dense_e
from abstract.T_diag_e import T_diag_e
from abstract.T_softabs import T_softabs_e
from abstract.T_softabs_diag import T_softabs_diag_e
from abstract.T_softabs_diag_outer_product import T_softabs_diag_outer_product_e
from abstract.T_softabs_outer_product import T_softabs_outer_product
from time_diagnostics import time_diagnositcs
from explicit.genleapfrog_ult_util import eigen

class Hamiltonian(object):
    # hamiltonian function
    def __init__(self,V,metric):
        self.V = V
        self.metric = metric
        if self.metric.name=="unit_e":
            obj = T_unit_e(metric,self.V)

        elif self.metric.name=="diag_e":
            obj = T_diag_e(metric,self.V)

        elif self.metric.name=="dense_e":
            obj = T_dense_e(metric,self.V)

        elif self.metric.name=="softabs":
            obj = T_softabs_e(metric,self.V)

        elif self.metric.name=="softabs_diag":
            obj = T_softabs_diag_e(metric,self.V)

        elif self.metric.name=="softabs_outer_product":
            obj = T_softabs_outer_product(metric,self.V)

        elif self.metric.name=="softabs_diag_outer_product":
            obj = T_softabs_diag_outer_product_e(metric,self.V)


        self.T = obj
        self.dG_dt = self.setup_dG_dt()
        self.p_sharp = self.setup_p_sharp()
        self.diagnostics = time_diagnositcs()
        self.V.diagnostics = self.diagnostics

    def evaluate_all(self,q_point=None,p_point=None):
        self.V.load_point(q_point)
        self.T.load_point(p_point)
        out = [0,self.V.evaluate_scalar(),self.T.evaluate_scalar()]
        out[0] = out[1]+out[2]
        self.diagnostics.add_num_H_eval(1)
        return(out)
    def evaluate(self,q_point=None,p_point=None):
        self.V.load_point(q_point)
        self.T.load_point(p_point)
        out = self.V.evaluate_scalar() + self.T.evaluate_scalar()
        #self.diagnostics.add_num_H_eval(1)
        return(out)

    def setup_dG_dt(self):
        if (self.metric.name == "softabs"):
            def dG_dt(q,p,dV=None,lam=None,Q=None,dH=None):
                if dV==None:
                    dV, H_, dH = self.V.getdH_tensor(q)
                    lam,Q = eigen(H_)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p.flattened_tensor, dH, Q, lam, alpha)) + self.V.dphidq(lam, alpha, dH, Q, dV))
        elif (self.metric.name == "softabs_diag"):
            def dG_dt(q, p,dV=None,mdiagH=None,mgraddiagH=None):
                if dV==None:
                    dV, mdiagH, mgraddiagH = self.V.getdH_diagonal_tensor(q)
                mlambda,_ = self.T.fcomputeMetric(mdiagH)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p.flattened_tensor,mdiagH,mlambda,mgraddiagH)) +
                        self.V.dphidq(mdiagH,mlambda))
        elif (self.metric.name=="softabs_diag_outer_product" ):
            def dG_dt(q, p,dV=None):
                if dV==None:
                    dV = self.V.getdV_tensor(q)
                mlambda, _ = self.T.fcomputeMetric(dV)
                mH = self.mH(dV)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p, dV, mlambda,mH)) +
                        self.V.dphidq(p.flattened_tensor,mlambda,dV,mH))
        elif (self.metric.name=="softabs_outer_product" ):
            def dG_dt(q, p,dV=None):
                if dV==None:
                    dV = self.V.getdV_tensor(q)
                    mH = self.mH(dV)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p.flattened_tensor,dV)) +
                        self.V.dphidq(dV,mH))
        else:
            def dG_dt(q,p):
                return (2 * self.T.evaluate(p) - torch.dot(q.flattened_tensor, self.V.dq(q)))

    def setup_p_sharp(self):
        if (self.metric == "softabs"):
            def p_sharp(q,p,lam=None,Q=None):
                out = p.point_clone()
                out.flattened_tensor = self.T.dtaudp(p.flattened_tensor,lam,Q)
                out.loadfromflatten()
                return(out)
        elif(self.metric.name=="softabs_diag"):
            def p_sharp(q,p,mlambda=None):
                out = p.point_clone()
                out.flattened_tensor = self.T.dtaudp(p.flattened_tensor,mlambda)
                out.loadfromflatten()
                return(out)
        elif(self.metric.name=="softabs_outer_product" or self.metric.name=="softabs_diag_outer_product"):
            def p_sharp(q,p,dV=None):
                out = p.point_clone()
                out.flattened_tensor = self.T.dtaudp(p.flattened_tensor,dV)
                out.loadfromflatten()
                return(out)

        else:
            def p_sharp(q,p):
                out = p.point_clone()
                out.flattened_tensor = self.T.dp(p.flattened_tensor)
                out.loadfromflatten()
                return(out)

        return(p_sharp)