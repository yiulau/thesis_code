import torch
from abstract_class_V import T
from explicit.genleapfrog_ult_util import eigen, softabs_map, coth_torch,J,D
class T_softabs_e(T):
    def __init__(self):
        super(T_softabs_e, self).__init__()
        return ()

    def evaluate_float(self):
        _, H_ = self.linkedV.getH()
        lam, Q = eigen(H_.data)
        temp = softabs_map(lam, self.metric.alpha)
        inv_exp_H = torch.mm(torch.mm(Q, torch.diag(1 / temp)), torch.t(Q))
        o = 0.5 * torch.dot(self.flattened_p_tensor, torch.mv(inv_exp_H, self.flattened_p_tensor))
        temp2 = 0.5 * torch.log((temp)).sum()
        output = o + temp2
        return (output)

    def dtaudp(self,p_flattened_tensor,lam=None,Q=None):
        if Q == None or lam == None:
            _, H_ = self.linkedV.getH_tensor()
            lam, Q = eigen(H_)
        return (Q.mv(torch.diag(1 / softabs_map(lam, self.metric.alpha)).mv((torch.t(Q).mv(p_flattened_tensor)))))

    def dtaudq(self,p_flattened_tensor,alpha=None,H_=None,dH=None):
        # returns flattened tensor
        if H_ == None or dH == None:
            _, H_, dH = self.linkedV.getdH_tensor()
            lam, Q = eigen(H_)
        alpha = self.metric.alpha
        N = self.dim
        Jm = J(lam, alpha, self.dim)
        Dm = D(p_flattened_tensor, Q, lam, alpha)
        M = torch.mm(Q, torch.mm(Dm, torch.mm(Jm, torch.mm(Dm, torch.t(Q)))))
        delta = torch.zeros(N)
        for i in range(N):
            delta[i] = 0.5 * torch.trace(-torch.mm(M, dH[i, :, :]))
        return (delta)

    def dphidq(self,lam, alpha, dH, Q, dV):
        # returns pytorch tensor
        N = len(lam)
        Jm = J(lam, alpha, len(lam))
        R = torch.diag(1 / (lam * coth_torch(alpha * lam)))
        M = torch.mm(Q, torch.mm(R * Jm, torch.t(Q)))
        delta = torch.zeros(N)
        for i in range(N):
            delta[i] = 0.5 * torch.trace(torch.mm(M, dH[i, :, :])) + dV[i]
        return (delta)

    def generate_momentum(self,lam=None,Q=None):
        if lam == None or Q == None:
            H_ = self.linkedV.getH()
            lam, Q = eigen(H_.data)
        temp = torch.mm(Q, torch.diag(torch.sqrt(softabs_map(lam, self.metric.alpha))))
        self.store_momentum.copy_(torch.mv(temp, torch.randn(len(lam))))
        return(self.store_momentum)