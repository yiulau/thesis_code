from abstract.mcmc_sampler import mcmc_sampler,mcmc_sampler_settings,initialization
from distributions.logistic_regression import V_logistic_regression
from experiments.experiment_obj import tuneinput_class
#from adapt_util.adapter_class import adapter_class
from experiments.experiment_obj import experiment
mcmc_meta1 = mcmc_sampler_settings(mcmc_id=0,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,
                                   warmup_per_chain=5,is_float=False,isstore_to_disk=False)

#print(mcmc_meta1.__dict__)

v_obj1 = V_logistic_regression()
input_dict = {"v_fun":[V_logistic_regression],"epsilon":["dual"],"second_order":[False],
              "evolve_L":[10],"metric_name":["unit_e"],"dynamic":[False],"windowed":[False],"criterion":[None]}



ep_dual_metadata_argument = {"name":"epsilon","target":0.65,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"max_second_per_sample":0.1,"obj_fun":"accept_rate","par_type":"fast"}




gpyopt_slow_metadata_argument = {"obj_fun":"ESJD","par_type":"slow","name":"gpyopt"}
gpyopt_medium_metadata_argument = {"obj_fun":"ESJD","par_type":"medium","name":"gpyopt"}
gpyopt_fast_metatdata_argument = {"obj_fun":"ESJD","par_type":"fast","name":"gpyopt"}
# shared by all chains in the same sampling session
fast_tune_setting_dict = {"epsilon":ep_dual_metadata_argument}
medium_tune_setting_dict = {"gpyopt":gpyopt_medium_metadata_argument}
slow_tune_setting_dict = {"gpyopt":gpyopt_slow_metadata_argument}

tune_settings_dict = {"fast":fast_tune_setting_dict,"medium":medium_tune_setting_dict,"slow":slow_tune_setting_dict}

# controls every tuning paramter for an mcmc_sampler object
def tuning_settings(list_arguments):
    out = {}
    fast_tune_setting_dict = {}
    medium_tune_setting_dict = {}
    slow_tune_setting_dict = {}
    for param in list_arguments:
        if param["par_type"]=="fast":
            fast_tune_setting_dict.update({param["name"]:param})
        if param["par_type"]=="medium":
            medium_tune_setting_dict.update({param["name"]:param})
        if param["par_type"]=="slow":
            slow_tune_setting_dict.update({param["name"]:param})
    out = tune_settings_dict = {"fast":fast_tune_setting_dict,"medium":medium_tune_setting_dict,"slow":slow_tune_setting_dict}

    return(out)
tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()

#print(tune_dict)

#exit()
#print(tuneable_obj.tune_param_grid.size)
#print(tuneable_obj.singleton_tune_dict())

#tune_dict = tuneable_obj.tune_param_grid
#print(tuneable_obj.tune_param_grid.shape)
#exit()
#initialization_obj = initialization(v_obj1)
sampler1 = mcmc_sampler(tune_dict,mcmc_meta1,tune_settings_dict)
sampler1.prepare_chains()

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