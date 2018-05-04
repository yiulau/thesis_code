# convert to stable implementation
import math,torch,numpy,time
def finite_1stderiv(f,h):

    out = f(+h)-f(-h)
    out = out/(2*h)
    #signout = 2*float(out>0)-1.
    #out = math.exp(math.log(abs(out)) - math.log(2*h))*signout
    return(out)


def finite_2ndderiv(f_2var,h):

    out = (f_2var(+h,+h) + f_2var(-h,-h)) - (f_2var(+h,-h) + f_2var(-h,+h))
    out = out/(4*h*h)

    return(out)

def finite_3rdderiv(f_3var,h):
    out = (f_3var(+h,+h,+h) + f_3var(+h,-h,-h) + f_3var(-h,+h,-h) +f_3var(-h,-h,+h))\
          -( f_3var(+h,-h,+h) + f_3var(+h,+h,-h) + f_3var(-h,+h,+h) + f_3var(-h,-h,-h))

    out = out/(8*h*h*h)
    return(out)



def finite_diff_grad(v_object,h=1e-5):
    cur_beta = v_object.beta.data.clone()
    dim = len(cur_beta)
    out = torch.zeros(dim)
    for i in range(dim):
        cur_vari = cur_beta[i]
        def fun_wrapped(diffi):
            v_object.beta.data.copy_(cur_beta)
            v_object.beta.data[i] = v_object.beta.data[i] + diffi
            temp = v_object.forward().data[0]
            v_object.beta.data.copy_(cur_beta)
            return(temp)
        out[i] = finite_1stderiv(fun_wrapped,h)
    return(out)


def finite_diff_hessian(v_object,h=1e-5):
    cur_beta = v_object.beta.data.clone()
    dim = len(cur_beta)
    out = torch.zeros(dim,dim)
    for i in range(dim):
        for j in range(i,dim):
            def fun_wrapped(diffi,diffj):
                v_object.beta.data.copy_(cur_beta)
                v_object.beta.data[i]=v_object.beta.data[i]+diffi
                v_object.beta.data[j]=v_object.beta.data[j]+diffj
                temp = v_object.forward().data[0]
                v_object.beta.data.copy_(cur_beta)
                return(temp)

            out[i,j] = finite_2ndderiv(fun_wrapped,h)
            out[j,i] = out[i,j]
    return(out)


def finite_diff_dH(v_object,h=1e-2):
    cur_beta = v_object.beta.data.clone()
    dim = len(cur_beta)
    out = torch.zeros(dim, dim,dim)

    for i in range(dim):
        for j in range(dim):
            for k in range(j,dim):
                def fun_wrapped(diffi,diffj,diffk):
                    v_object.beta.data.copy_(cur_beta)
                    v_object.beta.data[i] = v_object.beta.data[i]+diffi
                    v_object.beta.data[j]= v_object.beta.data[j]+diffj
                    v_object.beta.data[k] = v_object.beta.data[k] + diffk
                    temp = v_object.forward().data[0]
                    v_object.beta.data.copy_(cur_beta)
                    return(temp)
                out[i,j,k] = finite_3rdderiv(fun_wrapped,h)
                out[i,k,j] = out[i,j,k]
    return(out)



def compute_and_display_results(v_object,num_rep=10):
    num_rep = 3

    time_total_grad_explicit = 0
    time_total_grad_finite = 0
    time_total_grad_autograd = 0
    store_diff_grad_exact_finite = []
    store_diff_grad_autograd_finite = []
    store_diff_grad_autograd_exact = []

    time_total_H_explicit = 0
    time_total_H_finite = 0
    time_total_H_autograd = 0
    store_diff_H_exact_finite = []
    store_diff_H_autograd_finite = []
    store_diff_H_autograd_exact = []

    time_total_dH_explicit = 0
    time_total_dH_finite = 0
    time_total_dH_autograd = 0
    store_diff_dH_exact_finite = []
    store_diff_dH_autograd_finite = []
    store_diff_dH_autograd_exact = []
    for i in range(num_rep):
        v_object.beta.data.copy_(torch.randn(len(v_object.beta)))
        cur_beta = v_object.beta.data.clone()

        # compute grad three different ways and accumulate computation time
        time_temp = time.time()
        explicit_grad = v_object.load_explicit_gradient()
        time_total_grad_explicit += time.time() - time_temp
        time_temp = time.time()
        fin_diff_grad = finite_diff_grad(v_object)
        time_total_grad_finite += time.time() - time_temp
        time_temp = time.time()
        autograd_grad = v_object.getdV().data
        time_total_grad_autograd += time.time() - time_temp

        l2norm_diff1stderiv = torch.dot(explicit_grad - fin_diff_grad, explicit_grad - fin_diff_grad)
        store_diff_grad_exact_finite.append(l2norm_diff1stderiv)
        l2norm_diff1stderiv_autograd = torch.dot(autograd_grad - fin_diff_grad, autograd_grad - fin_diff_grad)
        store_diff_grad_autograd_finite.append(l2norm_diff1stderiv_autograd)
        l2norm_diff1stderiv_autograd_explicit = torch.dot(autograd_grad - explicit_grad, autograd_grad - explicit_grad)
        store_diff_grad_autograd_exact.append(l2norm_diff1stderiv_autograd_explicit)
        # compute Hessian three different ways and accumulate computation time

        time_temp = time.time()
        explicit_H = v_object.load_explicit_H()
        time_total_H_explicit += time.time() - time_temp
        time_temp = time.time()
        fin_diff_H = finite_diff_hessian(v_object)
        time_total_H_finite += time.time() - time_temp
        time_temp = time.time()
        autograd_H = v_object.getH()[1].data
        time_total_H_autograd += time.time() - time_temp

        l2norm_diff2ndderiv = ((explicit_H - fin_diff_H) * (explicit_H - fin_diff_H)).sum()
        store_diff_H_exact_finite.append(l2norm_diff2ndderiv)
        l2norm_diff2ndderiv_autograd = (
                (autograd_H - fin_diff_H) * (autograd_H - fin_diff_H)).sum()
        store_diff_H_autograd_finite.append(l2norm_diff2ndderiv_autograd)
        l2norm_diff2ndderiv_autograd_explicit = (
                (autograd_H - explicit_H) * (autograd_H - explicit_H)).sum()
        store_diff_H_autograd_exact.append(l2norm_diff2ndderiv_autograd_explicit)
        # compute dH three different ways and accumulate computation time

        time_temp = time.time()
        explicit_dH = v_object.load_explicit_dH()
        time_total_dH_explicit += time.time() - time_temp
        time_temp = time.time()
        fin_diff_dH = finite_diff_dH(v_object)
        time_total_dH_finite += time.time() - time_temp
        time_temp = time.time()
        autograd_dH = v_object.getdH()[2]
        time_total_dH_autograd += time.time() - time_temp

        l2norm_diff3rdderiv = ((explicit_dH - fin_diff_dH) * (explicit_dH - fin_diff_dH)).sum()
        store_diff_dH_exact_finite.append(l2norm_diff3rdderiv)
        # print("l2 norm difference between exact and finite diff for the dH {} ".format(l2norm_diff3rdderiv))

        l2norm_diff3rdderiv_autograd = ((autograd_dH - fin_diff_dH) * (autograd_dH - fin_diff_dH)).sum()
        store_diff_dH_autograd_finite.append(l2norm_diff3rdderiv_autograd)
        # print("l2 norm difference between autograd and finite diff for the dH {} ".format(l2norm_diff3rdderiv_autograd))

        l2norm_diff3rdderiv_autograd_explicit = ((autograd_dH - explicit_dH) * (autograd_dH - explicit_dH)).sum()
        store_diff_dH_autograd_exact.append(l2norm_diff3rdderiv_autograd_explicit)
        # print("l2 norm difference between autograd and exact diff for the dH {} ".format(
        #    l2norm_diff3rdderiv_autograd_explicit))

    print("results for gradient")

    print("average explicit time for grad {}".format(time_total_grad_explicit / num_rep))
    print("average finite time for grad{}".format(time_total_grad_finite / num_rep))
    print("average autograd time for grad {}".format(time_total_grad_autograd / num_rep))

    print("mean exact-finite diff for grad{}".format(numpy.mean(store_diff_grad_exact_finite)))
    print("mean autograd-finite diff for grad{}".format(numpy.mean(store_diff_grad_autograd_finite)))
    print("mean autograd-exact diff for grad{}".format(numpy.mean(store_diff_grad_autograd_exact)))

    print("results for H")
    print("average explicit time for H {}".format(time_total_H_explicit / num_rep))
    print("average finite time for H {}".format(time_total_H_finite / num_rep))
    print("average autograd time for H {}".format(time_total_H_autograd / num_rep))

    print("mean exact-finite diff for H {}".format(numpy.mean(store_diff_H_exact_finite)))
    print("mean autograd-finite diff for H {}".format(numpy.mean(store_diff_H_autograd_finite)))
    print("mean autograd-exact diff for H {}".format(numpy.mean(store_diff_H_autograd_exact)))

    print("results for dH")
    print("average explicit time for dH {}".format(time_total_dH_explicit / num_rep))
    print("average finite time for dH {}".format(time_total_dH_finite / num_rep))
    print("average autograd time for dH {}".format(time_total_dH_autograd / num_rep))

    print("mean exact-finite diff for dH {}".format(numpy.mean(store_diff_dH_exact_finite)))
    print("mean autograd-finite diff for dH {}".format(numpy.mean(store_diff_dH_autograd_finite)))
    print("mean autograd-exact diff for dH {}".format(numpy.mean(store_diff_dH_autograd_exact)))
