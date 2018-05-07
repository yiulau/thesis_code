import torch
def tuneable_param(dynamic,second_order,metric,criterion,input_time):
    if dynamic==True:
        if second_order==True:
            if criterion=="xhmc":
                out = ("ep","xhmc_delta","alpha")
            elif criterion=="gnuts" or criterion=="nuts":
                out = ("ep","alpha")
            else:
                raise ValueError("unknown criterion")
        else:
            if criterion=="xhmc":
                if metric=="unit_e":
                    out = ("ep","xhmc_delta")
                elif metric=="diag_e":
                    out = ("ep","xhmc_delta","diag_cov")
                elif metric=="dense_e":
                    out = ("ep","xhmc_delta","cov")
                else:
                    raise ValueError("unknown metric")
            elif criterion=="gnuts" or criterion=="nuts":
                if metric=="unit_e":
                    out = ("ep")
                elif metric=="diag_e":
                    out = ("ep","diag_cov")
                elif metric=="dense_e":
                    out = ("ep","cov")
                else:
                    raise ValueError("unknown metric")

    else:
        if second_order==True:
            if input_time:
                out = ("ep","t","alpha")
            else:
                out = ("ep","L","alpha")

        else:
            if input_time:
                if metric == "unit_e":
                    out = ("ep","t")
                elif metric == "diag_e":
                    out = ("ep","t","diag_cov")
                elif metric == "dense_e":
                    out = ("ep","L","cov")
                else:
                    raise ValueError("unknown metric")
            else:
                if metric=="unit_e":
                    out = ("ep","L")
                elif metric=="diag_e":
                    out = ("ep","L","diag_cov")
                elif metric=="dense_e":
                    out = ("ep","L","cov")
                else:
                    raise ValueError("unknown metric")
    return(out)

def welford_tensor(next_sample,sample_counter,m_,m_2,diag):
    # next_sample pytorch tensor
    # used for calculating ave second per leapfrog
    # keep accumulating variance for monitoring purposes

    delta = (next_sample-m_)
    m_ += delta/sample_counter
    # torch.ger(x,y) = x * y^T
    if diag:
        m_2 += (next_sample-m_) * delta
    else:
        m_2 += torch.ger((next_sample-m_),delta)
    return(m_,m_2,sample_counter)
