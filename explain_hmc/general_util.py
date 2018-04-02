import torch, math


def logsumexp_torch(a,b):
    # stable way to calculate logsumexp
    # input torch tensor
    # output torch tensor = log(exp(a)+exp(b))

    s = torch.max(a,b)
    out = s + torch.log(torch.exp(a-s) + torch.exp(b-s))
    return(out)

def logsumexp(a, b):
    # stable way to calculate logsumexp
    # input float
    # output float = log(exp(a)+exp(b))
    s = max(a,b)
    output = s + math.log((math.exp(a-s) + math.exp(b-s)))
    return(output)