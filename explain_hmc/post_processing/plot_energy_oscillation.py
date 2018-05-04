import numpy
import matplotlib.pyplot as plt

# need to plot the two functions on the same graph
# need to save the graph

def plot_V_T(V_vec,T_vec,epsilon):
    # V_vec and T_vec comes from one long leapfrog/ genleapfrog trajectory. No acceptance or anything

    H_vec = V_vec + T_vec
    L = len(V_vec)
    t_vec = [0]*L
    for n in range(L):
        t = epsilon * n
        t_vec[n] = t

    x = t_vec

    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111)
    ax.plot(t_vec, V_vec, c='k', ls='-', label='V')
    ax.plot(t_vec, T_vec, c='k', marker="+", ls=':', label='T')
    ax.plot(x, H_vec, c='g', marker=(8, 2, 0), ls='--', label='H')
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.title('ep ={}, L ={}'.format(epsilon,len(H_vec)))
    plt.legend(loc=2)
    plt.draw()
    plt.show()
    plt.savefig('foo.png')
    return()






