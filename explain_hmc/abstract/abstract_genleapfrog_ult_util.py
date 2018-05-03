import torch, numpy, math
from explicit.genleapfrog_ult_util import eigen
from time_diagnostics import time_diagnositcs

# move to abstract_class_V
# dphidq = dVdq + log(det(Sigma(q)))




def genleapfrog_wrap(delta,H):
    def inside(q,p,ep,pi):
        return generalized_leapfrog(q,p,ep,delta,H)
    return(inside)

def generalized_leapfrog(q,p,epsilon,Ham,delta=0.1):
    # input output point object
    # can take anything but should output tensor

    dV,H_,dH = Ham.V.getdH_tensor(q)
    lam, Q = eigen(H_)



    # dphidq outputs and inputs takes flattened gradient in flattened form
    p.flattened_tensor -= epsilon * 0.5 * Ham.T.dphidq(lam,dH,Q,dV)



    p.load_flatten()
    rho = p.flattened_tensor.clone()
    pprime = p.flattened_tensor.clone()
    deltap = delta + 0.5
    count = 0



    while (deltap > delta) and (count < 5):
        # dtaudq returns gradient in flattened form
        pprime.copy_(rho - epsilon * 0.5 * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam))
        deltap = torch.max(torch.abs(p.flattened_tensor-pprime))

        count = count + 1
    if deltap>delta:
        first_fi_divergent = True
        return (q, p, True)
    else:
        first_fi_divergent = False
        p.flattened_tensor.copy_(pprime)
        p.load_flatten()

    #print(first_fi_divergent)
    #print(p.flattened_tensor)
    sigma = q.point_clone()
    qprime = q.flattened_tensor.clone()
    deltaq = delta + 0.5

    _,H_ = Ham.V.getH_tensor(sigma)
    olam,oQ = eigen(H_)
    count = 0


    while (deltaq > delta) and (count < 5):
        _,H_ = Ham.V.getH_tensor(q)
        lam,Q = eigen(H_)
        qprime = sigma.flattened_tensor + 0.5 * epsilon * Ham.T.dtaudp(p.flattened_tensor,olam,oQ) + \
                 0.5 * epsilon* Ham.T.dtaudp(p.flattened_tensor,lam,Q)
        deltaq = torch.max(torch.abs(q.flattened_tensor-qprime))
        q.flattened_tensor.copy_(qprime)
        q.load_flatten()
        count = count + 1
    if deltaq>delta:
        second_fi_divergent = True
        return(q,p,True)
    else:
        second_fi_divergent = False
        q.flattened_tensor.copy_(qprime)
        q.load_flatten()
    print("H is {}".format(Ham.evaluate(q,p)))


    dV,H_,dH = Ham.V.getdH_tensor(q)
    lam,Q = eigen(H_)
    from explicit.genleapfrog_ult_util import softabs_map
    print(0.5 * epsilon * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam))

    p.flattened_tensor -= 0.5 * epsilon * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam)
    print("H is {}".format(Ham.evaluate(q, p)))
    p.load_flatten()

    p.flattened_tensor -= 0.5 * epsilon * Ham.T.dphidq(lam,dH,Q,dV)

    p.load_flatten()

    return(q,p,False)

def generalized_leapfrog_softabsdiag(q,p,epsilon,delta,Ham):
    # input output point object
    # can take anything but should output tensor
    dV,mdiagH,mgraddiagH = Ham.V.get_graddiagH(q)
    mlambda,_ = Ham.T.fcomputeMetric(mdiagH)
    # dphidq outputs and inputs takes flattened gradient in flattened form
    p.flattened_tensor -= epsilon * 0.5 * Ham.T.dphidq(dV,mdiagH,mgraddiagH,mlambda)
    p.load_flatten()
    rho = p.flattened_tensor.clone()
    pprime = p.flattened_tensor.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        # dtaudq returns gradient in flattened form
        pprime = rho - epsilon * 0.5 * Ham.T.dtaudq(p.flattened_tensor,mdiagH,mlambda,mgraddiagH)
        deltap = torch.max(torch.abs(p.flattened_tensor-pprime))
        p.flattened_tensor.copy_(pprime)
        p.load_flatten()
        count = count + 1

    sigma = q.point_clone()
    qprime = q.flattened_tensor.clone()
    deltaq = delta + 0.5

    _,mdiagH = Ham.V.getdiagH_tensor(sigma)
    omlambda,_ = Ham.T.fcomputeMetric(mdiagH)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,mdiagH = Ham.V.getdiagH_tensor(q)
        mlambda,_ = Ham.T.fcomputeMetric(mdiagH)
        qprime = sigma.flattened_tensor + 0.5 * epsilon * Ham.T.dtaudp(p.flattened_tensor,omlambda) + \
                 0.5 * epsilon* Ham.T.dtaudp(p.flattened_tensor,mlambda)
        deltaq = torch.max(torch.abs(q.flattened_tensor-qprime))
        q.flattened_tensor.copy_(qprime)
        q.load_flatten()
        count = count + 1

    dV, mdiagH, mgraddiagH = Ham.V.get_graddiagH(q)
    mlambda, _ = Ham.T.fcomputeMetric(mdiagH)

    p.flattened_tensor -= 0.5 * epsilon * Ham.T.dtaudq(p.flattened_tensor,mdiagH,mlambda,mgraddiagH)
    p.load_flatten()
    p.flattened_tensor -=0.5 * epsilon * Ham.T.dphidq(dV,mdiagH,mgraddiagH,mlambda)
    p.load_flatten()

    return(q,p)

def generalized_leapfrog_softabs_op(q,p,epsilon,delta,Ham):
    # input output point object
    # can take anything but should output tensor
    dV,H_,dH = Ham.V.getdH_tensor(q)
    lam, Q = eigen(H_)
    # dphidq outputs and inputs takes flattened gradient in flattened form
    p.flattened_tensor -= epsilon * 0.5 * Ham.T.dphidq(lam,dH,Q,dV)
    p.load_flatten()
    rho = p.flattened_tensor.clone()
    pprime = p.flattened_tensor.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        # dtaudq returns gradient in flattened form
        pprime = rho - epsilon * 0.5 * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam)
        deltap = torch.max(torch.abs(p.flattened_tensor-pprime))
        p.flattened_tensor.copy_(pprime)
        p.load_flatten()
        count = count + 1

    sigma = q.point_clone()
    qprime = q.flattened_tensor.clone()
    deltaq = delta + 0.5

    _,H_ = Ham.V.getH_tensor(sigma)
    olam,oQ = eigen(H_)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = Ham.V.getH_tensor(q)
        lam,Q = eigen(H_)
        qprime = sigma.flattened_tensor + 0.5 * epsilon * Ham.T.dtaudp(p.flattened_tensor,olam,oQ) + \
                 0.5 * epsilon* Ham.T.dtaudp(p.flattened_tensor,lam,Q)
        deltaq = torch.max(torch.abs(q.flattened_tensor-qprime))
        q.flattened_tensor.copy_(qprime)
        q.load_flatten()
        count = count + 1

    dV,H_,dH = Ham.V.getdH_tensor(q)
    lam,Q = eigen(H_)

    p.flattened_tensor -= 0.5 * epsilon * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam)
    p.load_flatten()
    p.flattened_tensor -=0.5 * epsilon * Ham.T.dphidq(lam,dH,Q,dV)
    p.load_flatten()

    return(q,p)

def generalized_leapfrog_softabs_op_diag(q,p,epsilon,delta,Ham):
    # input output point object
    # can take anything but should output tensor
    dV,H_,dH = Ham.V.getdH_tensor(q)
    lam, Q = eigen(H_)
    # dphidq outputs and inputs takes flattened gradient in flattened form
    p.flattened_tensor -= epsilon * 0.5 * Ham.T.dphidq(lam,dH,Q,dV)
    p.load_flatten()
    rho = p.flattened_tensor.clone()
    pprime = p.flattened_tensor.clone()
    deltap = delta + 0.5
    count = 0
    while (deltap > delta) and (count < 5):
        # dtaudq returns gradient in flattened form
        pprime = rho - epsilon * 0.5 * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam)
        deltap = torch.max(torch.abs(p.flattened_tensor-pprime))
        p.flattened_tensor.copy_(pprime)
        p.load_flatten()
        count = count + 1

    sigma = q.point_clone()
    qprime = q.flattened_tensor.clone()
    deltaq = delta + 0.5

    _,H_ = Ham.V.getH_tensor(sigma)
    olam,oQ = eigen(H_)
    count = 0
    while (deltaq > delta) and (count < 5):
        _,H_ = Ham.V.getH_tensor(q)
        lam,Q = eigen(H_)
        qprime = sigma.flattened_tensor + 0.5 * epsilon * Ham.T.dtaudp(p.flattened_tensor,olam,oQ) + \
                 0.5 * epsilon* Ham.T.dtaudp(p.flattened_tensor,lam,Q)
        deltaq = torch.max(torch.abs(q.flattened_tensor-qprime))
        q.flattened_tensor.copy_(qprime)
        q.load_flatten()
        count = count + 1

    dV,H_,dH = Ham.V.getdH_tensor(q)
    lam,Q = eigen(H_)

    p.flattened_tensor -= 0.5 * epsilon * Ham.T.dtaudq(p.flattened_tensor,dH,Q,lam)
    p.load_flatten()
    p.flattened_tensor -=0.5 * epsilon * Ham.T.dphidq(lam,dH,Q,dV)
    p.load_flatten()

    return(q,p)
def rmhmc_step(initq,epsilon,L,delta,Ham,careful=True):
    #q = Variable(initq.data.clone(), requires_grad=True)

    Ham.diagnostics = time_diagnositcs()
    q = initq.point_clone()
    #_, H_ = Ham.V.getH_tensor(q)
    #lam, Q = eigen(H_)
    #p = Variable(generate_momentum(alpha,lam,Q))
    p = Ham.T.generate_momentum(q)

    current_H = Ham.evaluate(q,p)

    if Ham.T.metric.name=="softabs_e":
        integrator = generalized_leapfrog
    elif Ham.T.metric.name=="softabs_diag":
        integrator = generalized_leapfrog_softabsdiag
    elif Ham.T.metric.name=="softabs_outer_product":
        integrator = generalized_leapfrog_softabs_op
    elif Ham.T.metric.name=="softabs_outer_product_diag":
        integrator = generalized_leapfrog_softabs_op_diag
    else:
        raise ValueError("not one of available metrics")
    for i in range(L):
        out = integrator(q,p,epsilon,delta,Ham)
        q = out[0]
        p = out[1]
        if careful:
            temp_H = Ham.evaluate(q, p)
            if (abs(temp_H - current_H) > 1000):
                return_q = initq
                return_H = current_H
                accept_rate = 0
                accepted = False
                divergent = True
                return (return_q, return_H, accepted, accept_rate, divergent,i)
        #q.flattened_tensor.copy_(out[0])
        #p.flattened_tensor.copy_(out[1])
        #q.load_flatten()
        #p.load_flatten()
    proposed_H = Ham.evaluate(q,p)
    u = numpy.random.rand(1)
    diff = current_H - proposed_H
    if (abs(diff) > 1000):
        divergent = True
    else:
        divergent = False
    accept_rate = math.exp(min(0,diff))
    if u < accept_rate:
        next_q = q
        accepted = True
    else:
        next_q = initq
        accepted = False
    return(next_q,proposed_H,accepted,accept_rate,divergent)