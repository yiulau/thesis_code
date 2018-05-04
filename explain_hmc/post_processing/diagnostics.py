import numpy

# bfmi-e

def bfmi_e(vector_of_energy):
    e_bar = numpy.mean(vector_of_energy)
    denom = numpy.square(vector_of_energy - e_bar).sum()
    numerator = numpy.square(vector_of_energy[1:]-vector_of_energy[:(len(vector_of_energy)-1)]).sum()
    out = numerator/denom
    return(out)


def lpd(p_y_given_theta,posterior_samples,observed_data):
    S = len(posterior_samples)
    n = len(observed_data)
    out = 0
    for i in range(n):
        temp = 0
        for j in range(S):
            temp +=p_y_given_theta(observed_data[i],posterior_samples[j])
        out += temp
    return(out)


def pwaic(log_p_y_given_theta,posterior_samples,observed_data):
    S = len(posterior_samples)
    n = len(observed_data)
    out = 0
    for i in range(n):
        temp = numpy.zeros(S)
        for j in range(S):
            temp[j]= log_p_y_given_theta(observed_data[i], posterior_samples[j])
        out += numpy.var(temp)
    return (out)

def WAIC(posterior_samples,observed_data,V):
    out = lpd(V.p_y_given_theta,posterior_samples,observed_data) - pwaic(V.log_p_y_given_theta,posterior_samples,observed_data)
    return(out)