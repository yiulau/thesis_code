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
    m_ = m_ + delta/sample_counter
    m_2 = m_2 + torch.ger((q-m_),delta)


empCov = numpy.cov(data.numpy(),rowvar=False)
emmean = numpy.mean(data.numpy(),axis=0)
print("numpy mean is {},welford mean is {}".format(emmean,m_))

print("numpy cov is {},welford cov is {}".format(empCov,m_2/(sample_counter-1)))
