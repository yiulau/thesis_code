import numpy,os

def mcse(numpy_vec):
    numpy.save("temp_mcse.npy", numpy_vec)
    os.system("R CMD BATCH mcse_in_R.R")
    fromr = numpy.load("temp_mcse_out.npy")
    #print(fromr)
    return(fromr[0])

def mcse_matrix(numpy_matrix):
    numpy.save("temp_mcse_matrix.npy", numpy_matrix)
    os.system("R CMD BATCH mcse_in_R_matrix.R")
    fromr = numpy.load("temp_mcse_matrix_out.npy")
    print(fromr)
    return(fromr)
#x=numpy.random.randn(10000,1000)

#mcse_matrix(x)
#print(mcse(x[:,0]))
#exit()
#for i in range(1000):
#    print("counter {}".format(i))
#    print(mcse(x[:, i]))
#print(x)
#mcse(x)

