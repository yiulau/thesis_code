import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy

# simple heat map
#x, y, z = np.loadtxt('data.txt', unpack=True)
#N = int(len(z)**.5)
#z = z.reshape(N, N)

#grid_len = 20

#x = numpy.random.randn(grid_len)
#y = numpy.random.randn(grid_len)*10
#z = numpy.random.randn(grid_len,grid_len)
#z = z.reshape(grid_len,grid_len) + 10
## important to set aspect to auto otherwise the scales wont adjust to give square image
#plt.imshow(z, extent=(numpy.amin(x), numpy.amax(x), numpy.amin(y), numpy.amax(y)),aspect="auto",
 #       cmap=cm.hot)
#plt.colorbar()
#plt.show()


# heatmap with two different types of dots and legends


grid_len = 20

x = numpy.random.randn(grid_len)*5
y = numpy.random.randn(grid_len)*10
z = numpy.random.randn(grid_len,grid_len)
z = z.reshape(grid_len,grid_len) + 10

N = 50
x_dots = numpy.random.rand(N)
y_dots = numpy.random.rand(N)

x_dots2 = numpy.random.randn(N)
y_dots2 = numpy.random.randn(N)
colors = numpy.random.rand(N)

# alpha controls transparency (0=transparent, 1 = opaque)
#plt.scatter(x_dots, y_dots, c="green",alpha=1)
plt.scatter(x_dots, y_dots,c="g",alpha=1,label="div")
plt.scatter(x_dots2,y_dots,c="r",alpha=1,label="div_H")
plt.legend()
plt.xlabel("theta")
plt.ylabel("x")
plt.title("how about that ")
#plt.scatter(x_dots2,y_dots,c="white",alpha=1)
# important to set aspect to auto otherwise the scales wont adjust to give square image
plt.imshow(z, extent=(numpy.amin(x), numpy.amax(x), numpy.amin(y), numpy.amax(y)),aspect="auto",
           cmap=plt.get_cmap("spring"))
        #cmap=cm.hot)
clb = plt.colorbar()
clb.ax.set_title('log scale')
plt.show()



# plot two heat maps side by side

N = 50
x_dots = numpy.random.rand(N)
y_dots = numpy.random.rand(N)

x_dots2 = numpy.random.randn(N)
y_dots2 = numpy.random.randn(N)
colors = numpy.random.rand(N)

plt.subplot(1,2,1)
# alpha controls transparency (0=transparent, 1 = opaque)
#plt.scatter(x_dots, y_dots, c="green",alpha=1)
plt.scatter(x_dots, y_dots,c="g",alpha=1,label="div")
plt.scatter(x_dots2,y_dots,c="r",alpha=1,label="div_H")
plt.legend()
plt.xlabel("theta")
plt.ylabel("x")
plt.title("plot2 ")
#plt.scatter(x_dots2,y_dots,c="white",alpha=1)
# important to set aspect to auto otherwise the scales wont adjust to give square image
plt.imshow(z, extent=(numpy.amin(x), numpy.amax(x), numpy.amin(y), numpy.amax(y)),aspect="auto",
           cmap=plt.get_cmap("spring"))
        #cmap=cm.hot)
clb = plt.colorbar()
clb.ax.set_title('log scale')



plt.subplot(1,2,2)

x_dots3 = numpy.random.rand(N)
y_dots3 = numpy.random.rand(N)

x_dots4 = numpy.random.randn(N)
y_dots4 = numpy.random.randn(N)

plt.scatter(x_dots, y_dots,c="g",alpha=1,label="div")
plt.scatter(x_dots2,y_dots,c="r",alpha=1,label="div_H")
plt.legend()
plt.xlabel("theta")
plt.ylabel("x")
plt.title("plot2 ")
#plt.scatter(x_dots2,y_dots,c="white",alpha=1)
# important to set aspect to auto otherwise the scales wont adjust to give square image
plt.imshow(z, extent=(numpy.amin(x), numpy.amax(x), numpy.amin(y), numpy.amax(y)),aspect="auto",
           cmap=plt.get_cmap("spring"))
        #cmap=cm.hot)
clb = plt.colorbar()
clb.ax.set_title('log scale')

plt.show()
