import numpy,torch,math
def get_objective_fun(fun_name):
    if fun_name=="accept_rate":
        out = average_accept_rate
    elif fun_name=="ESJD":
        out = ESJD
    elif fun_name=="ESJD_g_normalized":
        out = ESJD_g_normalized
    else:
        raise ValueError("unknown objective fun")
    return(out)

def average_accept_rate(store_samples):
    accept_rate_list = [None]*len(store_samples)
    for i in range(len(accept_rate_list)):
        accept_rate_list[i] = store_samples[i]["log"].store["accept_rate"]

    out = numpy.mean(accept_rate_list)
    return(out)

def ESJD(store_samples):

    diff_square_list = [None]*(len(store_samples)-1)
    for i in range(len(diff_square_list)):
        q_next_flattened_tensor = store_samples[i+1]["q"].flattened_tensor
        q_current_flattened_tensor = store_samples[i]["q"].flattened_tensor
        diff_squared = (q_next_flattened_tensor-q_current_flattened_tensor)
        diff_squared = torch.dot(diff_squared,diff_squared)
        diff_square_list[i] = diff_squared

    out = numpy.mean(diff_square_list)
    return(out)


def ESJD_g_normalized(store_samples):
    diff_square_list = [None] * (len(store_samples) - 1)
    num_grad_list = [None]*(len(store_samples)-1)
    for i in range(len(diff_square_list)):
        q_next_flattened_tensor = store_samples[i + 1]["q"].flattened_tensor
        q_current_flattened_tensor = store_samples[i]["q"].flattened_tensor
        diff_squared = (q_next_flattened_tensor - q_current_flattened_tensor)
        diff_squared = torch.dot(diff_squared, diff_squared)
        diff_square_list[i] = diff_squared
        num_grad = store_samples[i]["log"].store["num_transitions"]
        num_grad_list[i] = num_grad
    ESJD = numpy.mean(diff_square_list)
    #print(num_grad_list)
    #exit()
    ave_num_grad = numpy.mean(num_grad_list)
    out = ESJD/math.sqrt(ave_num_grad)
    return(out)


