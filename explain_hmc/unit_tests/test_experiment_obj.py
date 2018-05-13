import torch
from distributions.logistic_regression import V_logistic_regression
import abc, numpy, pickle, os
from abstract.mcmc_sampler import mcmc_sampler_settings,mcmc_sampler
from experiments.experiment_obj import tuneinput_class, experiment

#input_dict = {"v_fun":[V_logistic_regression],"alpha":[0],"epsilon":[0.1,0.2,0.3],"second_order":[False],"Cov":[torch.zeros(2)]}

input_dict = {"v_fun":[V_logistic_regression],"epsilon":[0.1,0.05],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}


input_obj = tuneinput_class(input_dict)
#print(input_obj.__dict__["grid_shape"])
#exit()
exper_obj = experiment(input_object=input_obj)

exper_obj.run()
#print(len(exper_obj.id_to_multi_index))

print(exper_obj.id_to_multi_index[0])
print(exper_obj.id_to_multi_index[1])

print(exper_obj.store_grid_obj[exper_obj.id_to_multi_index[0]])
print(exper_obj.store_grid_obj[exper_obj.id_to_multi_index[1]])

#exit()
#print(exper_obj.store_grid_obj[0,0,0,0,0])
#out.input_dict["v_fun"][0]()
#print()

#print(hex(id(out.Cov[0])))
#out2 = out.clone()

#print(hex(id(out2.Cov[0])))

#print(out.Cov=="adapt")
#exit()
#print(out.tune_param_grid)



