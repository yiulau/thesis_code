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

def stable_sum(a1,logw1,a2,logw2):
    # output sum_a =  (w1 * a1 + w2 * a2)/(w2 + w1)
    # log_sum = log(w1 + w2)
    if (logw2 > logw1):
        e = math.exp(logw1-logw2)
        sum_a = (e * a1 + a2)/(1.+e)
        log_sum = logw2 + math.log(1. + e)
    else:
        e = math.exp(logw2 - logw1)
        sum_a = (e * a2 + a1)/(1.+e)
        log_sum = logw1 + math.log(1. + e)
    return(sum_a,log_sum)

