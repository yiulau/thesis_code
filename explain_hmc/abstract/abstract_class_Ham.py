from abstract.T_unit_e import T_unit_e
from abstract.T_dense_e import T_dense_e
from abstract.T_diag_e import T_diag_e
from abstract.T_softabs import T_softabs_e
from abstract.T_softabs_diag import T_softabs_diag_e
from abstract.T_softabs_diag_outer_product import T_softabs_diag_outer_product_e
from abstract.T_softabs_outer_product import T_softabs_outer_product
from abstract.abstract_genleapfrog_ult_util import *
from abstract.abstract_leapfrog_ult_util import abstract_leapfrog_ult,windowerize
class Hamiltonian(object):
    # hamiltonian function
    def __init__(self,V,metric):
        self.V = V
        self.metric = metric
        if self.metric.name=="unit_e":
            T_obj = T_unit_e(metric,self.V)
            self.integrator = abstract_leapfrog_ult
        elif self.metric.name=="diag_e":
            T_obj = T_diag_e(metric,self.V)
            self.integrator = abstract_leapfrog_ult
        elif self.metric.name=="dense_e":
            T_obj = T_dense_e(metric,self.V)
            self.integrator = abstract_leapfrog_ult
        elif self.metric.name=="softabs":
            T_obj = T_softabs_e(metric,self.V)
            self.integrator = generalized_leapfrog
        elif self.metric.name=="softabs_diag":
            T_obj = T_softabs_diag_e(metric,self.V)
            self.integrator = generalized_leapfrog_softabsdiag
        elif self.metric.name=="softabs_outer_product":
            T_obj = T_softabs_outer_product(metric,self.V)
            self.integrator = generalized_leapfrog_softabs_op
        elif self.metric.name=="softabs_diag_outer_product":
            T_obj = T_softabs_diag_outer_product_e(metric,self.V)
            self.integrator = generalized_leapfrog_softabs_op_diag

        self.windowed_integrator = windowerize(self.integrator)

        self.T = T_obj
        self.dG_dt = self.setup_dG_dt()
        self.p_sharp_fun = self.setup_p_sharp()
        self.diagnostics = time_diagnositcs()
        self.V.diagnostics = self.diagnostics


    def evaluate_all(self,q_point=None,p_point=None):
        # returns (H,V,T)
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

            def dG_dt(q,p):
                dV, H_, dH = self.V.getdH_tensor(q)
                lam,Q = eigen(H_)
                out = -torch.dot(q.flattened_tensor, self.T.dtaudq(p.flattened_tensor, dH, Q, lam) + self.T.dphidq(lam, dH, Q, dV))

                return (out)
        elif (self.metric.name == "softabs_diag"):
            def dG_dt(q, p):

                dV, mdiagH, mgraddiagH = self.V.getdH_diagonal_tensor(q)
                mlambda,_ = self.T.fcomputeMetric(mdiagH)
                return (-torch.dot(q.flattened_tensor, self.T.dtaudq(p.flattened_tensor,mdiagH,mlambda,mgraddiagH)) +
                        self.T.dphidq(mdiagH,mlambda))
        elif (self.metric.name=="softabs_diag_outer_product"):
            def dG_dt(q, p):

                dV = self.V.getdV_tensor(q)
                mlambda, _ = self.T.fcomputeMetric(dV)
                mH = self.mH(dV)
                return (-torch.dot(q.flattened_tensor, self.T.dtaudq(p, dV, mlambda,mH)) +
                        self.T.dphidq(p.flattened_tensor,mlambda,dV,mH))
        elif (self.metric.name=="softabs_outer_product" ):
            def dG_dt(q, p):
                dV = self.V.getdV_tensor(q)
                mH = self.mH(dV)
                return (-torch.dot(q.flattened_tensor, self.T.dtaudq(p.flattened_tensor,dV)) +
                        self.T.dphidq(dV,mH))
        elif(self.metric.name=="dense_e" or self.metric.name=="diag_e" or self.metric.name=="unit_e"):
            def dG_dt(q,p):
                self.T.load_point(p)
                return (2 * self.T.evaluate_scalar() - torch.dot(q.flattened_tensor, self.V.dq(q.flattened_tensor)))
        else:
            raise ValueError("unknown metric name")
        return(dG_dt)
    def setup_p_sharp(self):
        if (self.metric.name == "softabs"):
            def p_sharp(q,p):
                _,H = self.V.getH_tensor(q)
                lam,Q = eigen(H)
                out = p.point_clone()
                out.flattened_tensor = self.T.dtaudp(p.flattened_tensor,lam,Q)
                out.load_flatten()
                return(out)
        elif(self.metric.name=="softabs_diag"):
            def p_sharp(q,p):
                _,mdiagH = self.V.getdiagH_tensor()
                mlambda,_ = self.T.fcomputeMetric(mdiagH)
                out = p.point_clone()
                out.flattened_tensor = self.T.dtaudp(p.flattened_tensor,mlambda)
                out.load_flatten()
                return(out)
        elif(self.metric.name=="softabs_outer_product" or self.metric.name=="softabs_diag_outer_product"):
            def p_sharp(q,p,dV=None):
                out = p.point_clone()
                out.flattened_tensor = self.T.dtaudp(p.flattened_tensor,dV)
                out.load_flatten()
                return(out)

        elif(self.metric.name=="dense_e" or self.metric.name=="unit_e" or self.metric.name=="diag_e"):
            def p_sharp(q,p):
                out = p.point_clone()
                out.flattened_tensor = self.T.dp(p.flattened_tensor)
                out.load_flatten()
                return(out)

        else:
            raise ValueError("unknown metric name")
        return(p_sharp)
