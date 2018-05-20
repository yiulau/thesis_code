from abstract.mcmc_sampler import mcmc_sampler,mcmc_sampler_settings,mcmc_sampler_settings_dict
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from experiments.experiment_obj import tuneinput_class
from adapt_util.tune_param_classes.tune_param_setting_util import *
import torch
mcmc_meta1 = mcmc_sampler_settings(mcmc_id=0,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=5,
                                   warmup_per_chain=1000,is_float=False,isstore_to_disk=False)
mcmc_meta2 = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=10000,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=5000,
                                   warmup_per_chain=1000,is_float=False,isstore_to_disk=False)


#print(mcmc_meta2)

#exit()
adapter_setting = default_adapter_setting()
#print(adapter_setting)
#exit()
#print(mcmc_meta1.__dict__)

#v_obj1 = V_logistic_regression()
#input_dict = {"v_fun":[V_logistic_regression],"epsilon":["dual"],"second_order":[False],
#              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}

input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"second_order":[False],"cov":["adapt"],"evolve_t":[2.3],
              "metric_name":["diag_e"],"dynamic":[False],"criterion":[None],"windowed":[False]}


# dual parameters input format
ep_dual_metadata_argument = {"name":"epsilon","target":0.65,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"obj_fun":"accept_rate","par_type":"fast"}
#ep_dual_metadata_argument = dual_default_arguments(name="epsilon")

#evolve_L_opt_metadata_argument = {"name":"evolve_L","obj_fun":"ESJD","bounds":(1,10),"par_type":"medium"}

#alpha_opt_metadata_argument = {"name":"alpha","obj_fun":"ESJD","par_type":"slow"}

#medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_L","alpha"],par_type="medium",bounds_list=[(1,10),(0.1,1e6)])
#medium_opt_metadata_argument = opt_default_arguments(name_list=["evolve_L","alpha"],par_type="medium")

# gpyopt parameters input format
#gpyopt_slow_metadata_argument = {"obj_fun":"ESJD","par_type":"slow","name":"gpyopt","params":("evolve_L","alpha")}
#gpyopt_medium_metadata_argument = {"obj_fun":"ESJD","par_type":"medium","name":"gpyopt","params":("evolve_t")}
#gpyopt_fast_metatdata_argument = {"obj_fun":"ESJD","par_type":"fast","name":"gpyopt"}



#dual_arguments = [ep_dual_metadata_argument,evolve_L_opt_metadata_argument,alpha_opt_metadata_argument]
dual_arguments = [ep_dual_metadata_argument]
#opt_arguments = [gpyopt_fast_metatdata_argument,gpyopt_medium_metadata_argument,gpyopt_slow_metadata_argument]
#opt_arguments = [medium_opt_metadata_argument]
opt_arguments = []
other_arguments = other_default_arguments()
adapt_cov_arguments = [adapt_cov_default_arguments(par_type="slow",dim=10)]
#
# tune_settings_dict = tuning_settings(dual_arguments,opt_arguments,other_arguments)
tuning_settings_dict = tuning_settings([],[],adapt_cov_arguments,[])

#print(tune_settings_dict)

# shared by all chains in the same sampling session
#fast_tune_setting_dict = {"epsilon":ep_dual_metadata_argument}
#medium_tune_setting_dict = {"gpyopt":gpyopt_medium_metadata_argument}
#slow_tune_setting_dict = {"gpyopt":gpyopt_slow_metadata_argument}

#tune_settings_dict = {"fast":fast_tune_setting_dict,"medium":medium_tune_setting_dict,"slow":slow_tune_setting_dict}

# controls every tuning paramter for an mcmc_sampler object

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
#print(tune_dict)

#print(tune_dict)


#print(tuneable_obj.tune_param_grid.size)
#print(tuneable_obj.singleton_tune_dict())

#tune_dict = tuneable_obj.tune_param_grid
#print(tuneable_obj.tune_param_grid.shape)
#exit()
#initialization_obj = initialization(v_obj1)
#sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_meta_obj=mcmc_meta1,tune_settings_dict=tune_settings_dict)
sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta2,tune_settings_dict=tuning_settings_dict)

sampler1.prepare_chains()

sampler1.pre_sampling_diagnostics()
exit()
#sampler1.prepare_chains()

out = sampler1.start_sampling()
exit()
print(out[0])
print(out[1])

print(sampler1.store_chains[0]["chain_obj"].store_samples)
exit()
#print(sampler1.store_chains[0]["chain_obj"].run())

out1= sampler1.run(0)
#out2= sampler1.run(1)

print(out1)
#print(out2)