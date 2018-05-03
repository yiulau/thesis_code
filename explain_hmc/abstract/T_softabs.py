import torch
from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
from explicit.genleapfrog_ult_util import eigen, softabs_map, coth_torch,J,D
# no updating should be done here
class T_softabs_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_softabs_e, self).__init__(linkedV)

    def evaluate_scalar(self):
        _, H_ = self.linkedV.getH_tensor()
        lam, Q = eigen(H_)

        temp = softabs_map(lam, self.metric.msoftabsalpha)

        inv_exp_H = torch.mm(torch.mm(Q, torch.diag(1/temp)), torch.t(Q))

        #print("inv_exp_H {}".format(inv_exp_H))
        o = 0.5 * torch.dot(self.flattened_tensor, torch.mv(inv_exp_H, self.flattened_tensor))
        temp2 = 0.5 * torch.log((temp)).sum()
        #print("alpha {}".format(self.metric.msoftabsalpha))
        #print("lam {}".format(lam))
        #print("H_ {}".format(H_))
        #print("H_2 {}".format(torch.mm(torch.mm(Q, torch.diag(temp)), torch.t(Q))))
        #print("msoftabslambda {}".format(temp))
        print("tau {}".format(o))
        print("logdetmetric {}".format(temp2))

        output = o + temp2
        return (output)

    def dtaudp(self,p_flattened_tensor,lam=None,Q=None):
        #if Q == None or lam == None:
        #    _, H_ = self.linkedV.getH_tensor()
        #    lam, Q = eigen(H_)
        return (Q.mv(torch.diag(1 / softabs_map(lam, self.metric.msoftabsalpha)).mv((torch.t(Q).mv(p_flattened_tensor)))))

    def dtaudq(self,p_flattened_tensor,dH,Q,lam):
        # returns flattened tensor
        #if H_ == None or dH == None:
        #    _, H_, dH = self.linkedV.getdH_tensor()
        #    lam, Q = eigen(H_)
        alpha = self.metric.msoftabsalpha
        N = self.dim
        Jm = J(lam, alpha, self.dim)
        print("J {}".format(Jm))
        print("lam {}".format(lam))
        Dm = D(p_flattened_tensor, Q, lam, alpha)
        print("D {}".format(Dm))
        M = torch.mm(Q, torch.mm(Dm, torch.mm(Jm, torch.mm(Dm, torch.t(Q)))))
        print("M {}".format(M))
        delta = torch.zeros(N)
        for i in range(N):
            delta[i] = 0.5 * torch.trace(-torch.mm(M, dH[i, :, :]))
        return (delta)

    def dphidq(self,lam, dH, Q, dV):
        # returns pytorch tensor
        alpha = self.metric.msoftabsalpha
        N = len(lam)
        Jm = J(lam, alpha, len(lam))
        R = torch.diag(1 / (lam * coth_torch(alpha * lam)))
        M = torch.mm(Q, torch.mm(R * Jm, torch.t(Q)))
        delta = torch.zeros(N)

        for i in range(N):
            delta[i] = 0.5 * torch.trace(torch.mm(M, dH[i, :, :])) + dV[i]
        return (delta)

    def generate_momentum(self,q):

        #if lam == None or Q == None:
        #    H_ = self.linkedV.getH_tensor()
        #    lam, Q = eigen(H_)
        _, H_ = self.linkedV.getH_tensor(q)
        lam, Q = eigen(H_)
        temp = torch.mm(Q, torch.diag(torch.sqrt(softabs_map(lam, self.metric.msoftabsalpha))))
        out = point(None,self)
        out.flattened_tensor.copy_(torch.mv(temp, torch.randn(len(lam))))
        out.load_flatten()
        return(out)