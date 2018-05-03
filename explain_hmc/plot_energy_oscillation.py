from distributions.banana_distribution import V_banana
from distributions.logistic_regression import V_logistic_regression
from distributions.funnel_cp import V_funnel
from distributions.eightschool_cp import V_eightschool_cp
from distributions.eightschool_ncp import V_eightschool_ncp
import numpy
import matplotlib.pyplot as plt
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_genleapfrog_ult_util import generalized_leapfrog,generalized_leapfrog_softabsdiag
from abstract.abstract_leapfrog_ult_util import abstract_leapfrog_ult
from abstract.abstract_nuts_util import abstract_NUTS
import torch
# need to plot the two functions on the same graph
# need to save the graph

#vo = V_banana()
#vo = V_logistic_regression()
vo = V_funnel()
#vo = V_eightschool_cp()
vo = V_eightschool_ncp()
metrico = metric("softabs",vo,alpha=1e6)
#metrico = metric("unit_e",vo)
seed=1
torch.manual_seed(seed)
numpy.random.seed(seed)
from abstract.T_unit_e import T_unit_e
#T_unit_e(metrico,vo)
#exit()
#to = T(metrico,vo)
Ho = Hamiltonian(vo,metrico)
initq = Ho.V.q_point
initq.flattened_tensor.copy_(torch.randn(10))
q = initq.point_clone()
q.load_flatten()
#print(q.flattened_tensor)

p = Ho.T.generate_momentum(q)

#print(p.flattened_tensor)

#current_H = Ho.evaluate(q,p)
L=100
epsilon = 0.1
delta = 0.1
Ham = Ho
H_list = []
V_list = []
T_list = []
#current_V,current_T,current_H = Ho.evaluate_all(q,p)
#H_list.append(current_H)
#V_list.append(current_V)
#T_list.append(current_T)
for _ in range(5000):
    #out = generalized_leapfrog_softabsdiag(q, p, epsilon,Ham,delta)
    out = generalized_leapfrog(q,p,epsilon,Ham,delta)
    #out = abstract_leapfrog_ult(q,p,epsilon,Ho)
    #out = abstract_NUTS(q,epsilon,Ham,abstract_leapfrog_ult,5)
    q = out[0]
    p = out[1]
    print("q {}".format(q.flattened_tensor))
    print("p {}".format(p.flattened_tensor))
    #print("first fi diverges {}".format(out[2]))
    #print("second fi diverges {}".format(out[3]))
    current_H,current_V, current_T = Ho.evaluate_all(q, p)
    print("current V is {}".format(current_V))
    print("current H {}".format(current_H))
    #current_V, current_T, current_H = Ho.evaluate_all(q, p)
    #print("current H2 {}".format(Ho.evaluate(q,p)))
    H_list.append(current_H)
    V_list.append(current_V)
    T_list.append(current_T)
    if out[2]:
        break
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






