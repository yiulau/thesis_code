import torch, numpy, math
from abstract.abstract_class_T import T
from abstract.abstract_class_point import point


class T_softabs_outer_product(T):
    def __init__(self):
        super(T_softabs_outer_product, self).__init__()
        return ()

    def evaluate_float(self,p=None,q=None,dV=None):
        gg = torch.dot(dV, dV)
        s = numpy.sinh(self.metric.msoftabsalpha * gg)
        c = numpy.cosh(self.metric.msoftabsalpha * gg)
        mLogdetmetric = - len(dV) * math.log(s / gg) + math.log(c)
        gv = torch.dot(dV, p)
        aux = (s / gg) * (p + (1 / c - 1) * (gv / gg) * dV)
        out = 0.5 * mLogdetmetric + 0.5 * torch.dot(p, p)
        return (out)

    def dtaudp(self,lam=None,Q=None,p_flattened_tensor=None,dV=None):
        msoftabsalpha = self.metric.alpha
        gg = torch.dot(dV, dV)
        gv = torch.dot(dV, p_flattened_tensor)
        s = numpy.sinh(self.metric.msoftabsalpha * gg).cast
        c = numpy.cosh(msoftabsalpha * gg).cast
        out = (s / gg) * (p_flattened_tensor + (1 / c - 1) * (gv / gg) * dV)
        return(out)

    def dtaudq(self,p_flattened_tensor=None,dV=None,mH=None):
        if mH==None:
            mH = self.mH(dV)
        gg = torch.dot(dV, dV)
        agg = self.metric.msoftabsalpha * gg
        gp = torch.dot(dV, p_flattened_tensor)
        pp = torch.dot(p_flattened_tensor, p_flattened_tensor)
        gpDgg = gp / gg
        s = numpy.sinh(agg)
        c = numpy.cosh(agg)
        t = s / c

        if agg < 1e-4:
            c1 = (1 / 3) * agg * agg
            c2 = 1.5 * agg * agg
            c3 = -0.5 * agg * agg
        else:
            c1 = c - s / agg
            c2 = c - 1 / (c * c)
            c3 = (t - s) / agg

        out = (2 * self.metric.msoftabsalpha * (c1 * (pp / gg) - (c2 + 2 * c3) * gpDgg * gpDgg)) * (torch.mv(mH, dV))
        out += (2 * self.metric.msoftabsalpha * c3 * gpDgg) * torch.mv(mH, p_flattened_tensor)

        return (out)

    def generate_momentum(self,lam=None,Q=None,dV=None):
        msoftabsalpha = self.metric.alpha
        gg = torch.dot(dV, dV)
        agg = msoftabsalpha * gg

        dV = dV * math.sqrt((numpy.cosh(agg) - 1) / gg)
        mH = torch.zeros(len(dV), len(dV))
        for i in range(len(dV)):
            v = dV[i]
            L = 1.
            r = math.sqrt(L * L + v * v)
            c = L / r
            s = v / r

            mH[i, i] = r
            for j in range(len(dV)):
                vprime = dV[j]
                Lprime = mH[i, j]

                dV[j] = c * vprime - s * Lprime
                mH[i, j] = s * vprime + c * Lprime

        mH = mH * math.sqrt(gg / numpy.sinh(agg))
        mHL = torch.potrf(mH,upper=False)
        out = point(None, self)
        out.flattened_tensor.copy_(torch.mv(mHL, torch.randn(len(dV))))
        out.load_flatten()
        return (out)

    def dphi_dq(self, dV=None,mH=None):
        if mH==None:
            mH = self.mH(dV)
        gg = torch.dot(dV, dV)
        agg = self.metric.msoftabsalpha * gg
        t = float(numpy.tanh(agg))
        out = torch.mv(mH, dV)

        if (abs(agg) < 1e-4):
            out = out * 2 * ((len(dV) / (3 * gg)) * agg * agg + self.metric.msoftabsalpha * t)
        else:
            out = out * 2((len(dV) / gg) * (1 - agg / t) + self.metric.msoftabsalpha * t)
        out = out + dV

        return (out)

    def mH(self,dV):
        msoftabsalpha = self.metric.alpha
        gg = torch.dot(dV, dV)
        agg = msoftabsalpha * gg

        dV = dV * math.sqrt((numpy.cosh(agg) - 1) / gg)
        mH = torch.zeros(len(dV), len(dV))
        for i in range(len(dV)):
            v = dV[i]
            L = 1.
            r = math.sqrt(L * L + v * v)
            c = L / r
            s = v / r

            mH[i, i] = r
            for j in range(len(dV)):
                vprime = dV[j]
                Lprime = mH[i, j]

                dV[j] = c * vprime - s * Lprime
                mH[i, j] = s * vprime + c * Lprime

        mH = mH * math.sqrt(gg / numpy.sinh(agg))
        return(mH)