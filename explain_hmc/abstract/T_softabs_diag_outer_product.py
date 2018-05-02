import torch, numpy, math
from abstract.abstract_class_T import T
from abstract.abstract_class_point import point

class T_softabs_diag_outer_product_e(T):
    def __init__(self):
        super(T_softabs_diag_outer_product_e, self).__init__()
        return ()

    def evaluate_float(self):
        dV = self.linkedV.getV()
        p = self.flattened_tensor
        mlambda, mlogdetmetric = self.fcomputemetric(dV)
        temp = mlambda * p
        out = 0.5 * torch.dot(temp, p) + 0.5 * mlogdetmetric
        return (out)

    def dtau_dp(self,dV, p_flattened_tensor):
        mlambda = torch.zeros(len(dV))
        for i in range(len(dV)):
            lam = dV[i] * dV[i]
            alphalambda = self.metric.msoftabsalpha * lam

            if (abs(alphalambda) < 1e-4):
                mlambda[i] = self.metric.msoftabsalpha * (1 - (1 / 3) * alphalambda * alphalambda)
            elif (abs(alphalambda) > 18):
                mlambda[i] = 1 / abs(alphalambda)

            else:
                mlambda[i] = numpy.tanh(self.metric.msoftabsalpha * lam) / lam

        return (mlambda * p_flattened_tensor)

    def dtau_dq(self,p_flattened_tensor=None,dV=None,mlambda=None,mH=None):
        if mlambda==None:
            mlambda,_ = self.fcomputemetric()
            mH = self.mH(dV, mlambda)
        mgradhelper = torch.zeros(len(dV))

        for i in range(len(dV)):
            g2 = dV[i] * dV[i]
            hlambda = g2 * mlambda[i]
            v = 1 / g2
            if (abs(self.metirc.msoftabsalpha * g2) < 18):
                v += self.metric.msoftabsalpha * (hlambda - 1 / hlambda)
                mgradhelper[i] = - mlambda[i] * p_flattened_tensor[i] * p_flattened_tensor[i] * dV[i] * v
        out = torch.mv(mH, mgradhelper)
        return (out)

    def generate_momentum(self,mlambda=None):
        if mlambda==None:
            mlambda,_ = self.fcomputemetric()
          # computed by fcomputemetric
        #out = torch.randn(len(self.dim)) / torch.sqrt(mlambda)
        out = point(None, self)
        out.flattened_tensor.copy_(torch.randn(len(self.dim)) / torch.sqrt(mlambda))
        out.load_flatten()
        return (out)

    def dphi_dq(self,p_flattened_tensor=None,mlambda=None,dV=None,mH=None):
        if dV==None:
            mlambda, _ = self.fcomputemetric()
            mH = self.mH(dV, mlambda)
        mgradhelper = torch.zeros(len(dV))
        for i in range(len(dV)):
            g2 = dV[i] * dV[i]
            hlambda = g2 * mlambda[i]
            v = 1 / g2
            if (abs(self.metric.msoftabsalpha * g2) < 18):
                v += self.metric.msoftabsalpha * (hlambda - 1 / hlambda)
                mgradhelper[i] = - mlambda[i] * p_flattened_tensor[i] * p_flattened_tensor[i] * dV[i] * v
        out = torch.mv(mH, mgradhelper) + dV
        return (out)

    def fcomputemetric(self,dV=None):
        mlambda = torch.zeros(len(dV))
        for i in range(len(dV)):
            lam = dV[i] * dV[i]
            alphalambda = self.metric.msoftabsalpha * lam

            if (abs(alphalambda) < 1e-4):
                mlambda[i] = self.metric.msoftabsalpha * (1 - (1 / 3) * alphalambda * alphalambda)
            elif (abs(alphalambda) > 18):
                mlambda[i] = 1 / abs(alphalambda)

            else:
                mlambda[i] = numpy.tanh(self.metric.msoftabsalpha * lam) / lam

        mlogdetmetric = 0
        for i in range(len(dV)):
            mlogdetmetric += math.log(mlambda[i])
        return (mlambda, mlogdetmetric)

    def mH(self,dV,mlambda):
        mH = torch.eye(len(dV),len(dV))
        for i in range(len(dV)):
            g2 = dV[i]*dV[i]
            hLambda = g2 * mlambda[i]
            v = 1./g2
            if(abs(self.metric.msoftabsalpha*g2)<20):
                v+= self.metric.msoftabsalpha * (hLambda -1./hLambda )

            mH[:,i]  = mH[:,i] * -2. * mlambda[i] * dV[i] * v
        return(mH)
