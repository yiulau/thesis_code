import numpy
from python2R.mcse_rpy2 import mcse_repy2 as mc_se
def check_mean_var(mcmc_samples,correct_mean,correct_cov,diag_only=False):
    # mcmc_samples a numpy matrix where each row is a sample
    # expects correct_cov to be a vector when diag_only = True
    numpy.mean(mcmc_samples, axis=0)
    #empCov = numpy.cov(mcmc_samples, rowvar=False)
    #emmean = numpy.mean(mcmc_samples, axis=0)
    #mc_se = mc_se(mcmc_samples)

    if diag_only:
        mcmc_Cov = numpy.empty(shape=[mcmc_samples.shape[1]],dtype=object)
    else:
        mcmc_Cov = numpy.empty(shape=[mcmc_samples.shape[1],mcmc_samples.shape[1]],dtype=object)
    mcmc_mean = numpy.empty(shape=(mcmc_samples.shape[1]),dtype=object)

    # first treat the means
    for i in range(len(mcmc_mean)):
        temp_vec = mcmc_samples[:,i]
        mu = numpy.mean(temp_vec)
        abs_diff = abs(mu - correct_mean[i])
        MCSE = mc_se(temp_vec)
        if abs_diff<3*MCSE:
            reasonable = True
        else:
            reasonable = False
        out = {"abs_diff":abs_diff,"MCSE":MCSE,"reasonable":reasonable}
        mcmc_mean[i] = out


    # treat the covariances
    if diag_only:
        for i in range(mcmc_Cov.shape[0]):
            temp_vec_i = mcmc_samples[:, i]
            var_temp_vec = numpy.square(temp_vec_i - correct_mean[i])
            mu = numpy.mean(var_temp_vec)
            MCSE = mc_se(var_temp_vec)
            abs_diff = abs(mu - correct_cov[i])
            if abs_diff < 3*MCSE:
                reasonable = True
            else:
                reasonable = False
            out = {"abs_diff": abs_diff, "MCSE": MCSE, "reasonable": reasonable}
            mcmc_Cov[i] = out
    else:
        for i in range(mcmc_Cov.shape[0]):
            for j in range(mcmc_Cov.shape[1]):
                if not i==j:
                    temp_vec_i = mcmc_samples[:,i]
                    temp_vec_j = mcmc_samples[:,j]
                    #covar_temp_vec = (temp_vec_i - correct_mean[i])*(temp_vec_j-correct_mean[j])/\
                    #                (numpy.sqrt(correct_cov[i,i]*correct_cov[j,j]))
                    covar_temp_vec = (temp_vec_i - correct_mean[i])*(temp_vec_j-correct_mean[j])
                    mu = numpy.mean(covar_temp_vec)
                    MCSE = mc_se(covar_temp_vec)
                    abs_diff = abs(mu-correct_cov[i,j])
                else:
                    temp_vec_i = mcmc_samples[:, i]
                    var_temp_vec = numpy.square(temp_vec_i - correct_mean[i])
                    mu = numpy.mean(var_temp_vec)
                    MCSE = mc_se(var_temp_vec)
                    abs_diff = abs(mu-correct_cov[i,i])
                if abs_diff < 3*MCSE:
                    reasonable = True
                else:
                    reasonable = False
                out = {"abs_diff":abs_diff,"MCSE":MCSE,"reasonable":reasonable}
                mcmc_Cov[i,j]=out

    denom = 0.
    num = 0.
    for i in range(len(mcmc_mean)):
        num +=float(mcmc_mean[i]["reasonable"])
        denom +=1
    pc_of_mean = num/denom
    num = 0.
    denom = 0.
    if diag_only:
        for i in range(mcmc_Cov.shape[0]):
            num += float(mcmc_Cov[i]["reasonable"])
            denom +=1
    else:
        for i in range(mcmc_Cov.shape[0]):
            for j in range(mcmc_Cov.shape[1]):
                num += float(mcmc_Cov[i,j]["reasonable"])
                denom +=1
    pc_of_cov = num/denom

    out = {"mcmc_mean":mcmc_mean,"mcmc_Cov":mcmc_Cov,"pc_of_mean":pc_of_mean,"pc_of_cov":pc_of_cov}
    return(out)





