import readline
import rpy2
#import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
import numpy
#pi = robjects.r['pi']

mcmcse = importr("mcmcse")
#ctl = FloatVector([4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14])
#out = mcmcse.mcse(ctl)
#print(out.names)
#print(type(out[1]))
#out = (numpy.asarray(out[1]))
#print(pi[0])

def mcse_repy2(numpy_vec):
    ctl = FloatVector(numpy_vec)
    out = mcmcse.mcse(ctl)
    out = numpy.asarray(out[1])
    return(out)

#x=numpy.random.randn(10000,1000)
#mcse_repy2(x[:,0])
#for i in range(1000):
#    print("counter {}".format(i))
#    print(mcse_repy2(x[:, i]))
#print(x)