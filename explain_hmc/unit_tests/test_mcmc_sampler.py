from abstract.mcmc_sampler import mcmc_sampler,mcmc_sampler_settings
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class

mcmc_meta1 = mcmc_sampler_settings(mcmc_id=0,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,
                                   warmup_per_chain=5,is_float=False,isstore_to_disk=False)

#print(mcmc_meta1.__dict__)

v_obj1 = V_logistic_regression()
input_dict = {"v_fun":[V_logistic_regression],"epsilon":[0.1],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}


tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
#print(tuneable_obj.tune_param_grid.size)
#print(tuneable_obj.singleton_tune_dict())

#tune_dict = tuneable_obj.tune_param_grid
#print(tuneable_obj.tune_param_grid.shape)
#exit()
#initialization_obj = initialization(v_obj1)
sampler1 = mcmc_sampler(tune_dict,mcmc_meta1)

#sampler1.prepare_chains()

out = sampler1.start_sampling()
print(out[0])
print(out[1])

print(sampler1.store_chains[0]["chain_obj"].store_samples)
exit()
#print(sampler1.store_chains[0]["chain_obj"].run())

out1= sampler1.run(0)
#out2= sampler1.run(1)

print(out1)
#print(out2)