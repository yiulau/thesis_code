from distributions.banana_distribution import V_banana
from distributions.logistic_regression import V_logistic_regression
import numpy
import matplotlib.pyplot as plt
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_genleapfrog_ult_util import generalized_leapfrog
from abstract.abstract_leapfrog_ult_util import abstract_leapfrog_ult
# need to plot the two functions on the same graph
# need to save the graph

#vo = V_banana()
vo = V_logistic_regression()
metrico = metric("softabs",vo,alpha=1e6)
#metrico = metric("unit_e",vo)

from abstract.T_unit_e import T_unit_e
#T_unit_e(metrico,vo)
#exit()
#to = T(metrico,vo)
Ho = Hamiltonian(vo,metrico)
initq = Ho.V.q_point
q = initq.point_clone()

p = Ho.T.generate_momentum(q)


#current_H = Ho.evaluate(q,p)
L=10
epsilon = 0.5
delta = 0.1
Ham = Ho
H_list = []
V_list = []
T_list = []
current_V,current_T,current_H = Ho.evaluate_all(q,p)
H_list.append(current_H)
V_list.append(current_V)
T_list.append(current_T)
for _ in range(L):
    out = generalized_leapfrog(q,p,epsilon,delta,Ham)
    #out = abstract_leapfrog_ult(q,p,epsilon,Ho)
    q = out[0]
    p = out[1]
    current_V, current_T, current_H = Ho.evaluate_all(q, p)
    H_list.append(current_H)
    V_list.append(current_V)
    T_list.append(current_T)
    if abs(current_H - H_list[0])>1000:
        break
    print(_)

print(V_list)
print(T_list)
print(H_list)
#exit()
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
    plt.title('ep ={}, L ={}'.format(epsilon,len(H_list)))
    plt.legend(loc=2)
    plt.draw()
    plt.show()
    plt.savefig('foo.png')
    return()


#V_vec = numpy.random.randn(200)
#T_vec = numpy.random.randn(200)
#epsilon = 0.1
V_vec = numpy.array(V_list)
T_vec = numpy.array(T_list)

plot_V_T(V_vec,T_vec,epsilon)






