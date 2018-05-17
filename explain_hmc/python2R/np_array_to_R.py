import numpy

x = numpy.random.randn(8,1400,4)
print(type(list(x.shape)[0]))

shape = numpy.array(list(x.shape))*1.
numpy.save("shape.npy",shape)
#print(x.shape[0])
#print(x)
for i in range(x.shape[0]):
    numpy.save("temp{}.npy".format(i), x[i,:,:])

# write function to delete shape.npy and temp*.npy files after being done with them

