import torch
import numpy

dim = 3
num_samples = 10
torch.set_printoptions(precision=10)
data = torch.randn((num_samples,dim))
#print(data[0,])
#exit()
m_ = torch.zeros(dim)
m_2 = torch.zeros((dim,dim))
sample_counter = 0
for i in range(num_samples):
    sample_counter = sample_counter + 1
    q = data[i,]
    delta = (q-m_)
    m_ +=  delta/sample_counter
    # torch.ger(x,y) = x * y^T
    m_2 += torch.ger((q-m_),delta)

def welford(next_sample,sample_counter,m_,m_2,diag):
    # next_sample pytorch tensor
    # diag boolean variable if true m_2 is the accumulative varinces
    #                       if false m_2 is the accumulative covars
    sample_counter = sample_counter + 1
    delta = (next_sample-m_)
    m_ += delta/sample_counter
    # torch.ger(x,y) = x * y^T
    if diag:
        m_2 += (next_sample-m_) * delta
    else:
        m_2 += torch.ger((next_sample-m_),delta)
    return(m_,m_2,sample_counter)

m_ = torch.zeros(dim)
#m_2 = torch.zeros((dim,dim))
m_2 = torch.zeros(dim)
sample_counter = 0
for i in range(num_samples):
    m_,m_2,sample_counter = welford(data[i,],sample_counter,m_,m_2,True)

empCov = numpy.cov(data.numpy(),rowvar=False)
emmean = numpy.mean(data.numpy(),axis=0)
print("numpy mean is {},welford mean is {}".format(emmean,m_))

print("numpy cov is {},welford cov is {}".format(empCov,m_2/(sample_counter-1)))
