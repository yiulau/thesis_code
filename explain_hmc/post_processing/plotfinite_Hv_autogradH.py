import matplotlib.pyplot as plt
import numpy
N = 200
t = 2
x_pts = numpy.linspace(0,t,N)
y1 = numpy.random.randn(N)
y2 = numpy.random.randn(N)

fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.plot(list(range(N)), y1, c='r', label='dletaH1')
ax.plot(list(range(N)), y2, c='g', label='deltaH2')
plt.xlabel('t')
plt.ylabel('energy')
plt.title('ep ={}, L ={}'.format(1,len(y1)))
plt.legend(loc=2)
plt.draw()
plt.show()
#plt.savefig('foo.png')