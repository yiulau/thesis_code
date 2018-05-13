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


# dual parameters input format
ep_dual_metadata_argument = {"name":"epsilon","target":0.65,"gamma":0.05,"t_0":10,
                        "kappa":0.75,"max_second_per_sample":0.1,"obj_fun":"accept_rate","par_type":"fast"}


evolve_L_opt_metadata_argument = {"name":"evolve_L","obj_fun":"ESJD","bounds":(1,10),"par_type":"medium"}

alpha_opt_metadata_argument = {"name":"alpha","obj_fun":"ESJD","par_type":"slow"}

# gpyopt parameters input format
gpyopt_slow_metadata_argument = {"obj_fun":"ESJD","par_type":"slow","name":"gpyopt","params":("evolve_L","alpha")}
gpyopt_medium_metadata_argument = {"obj_fun":"ESJD","par_type":"medium","name":"gpyopt","params":("evolve_t")}
gpyopt_fast_metatdata_argument = {"obj_fun":"ESJD","par_type":"fast","name":"gpyopt"}

def dual_default_arguments(name):
    output ={"name":name,"target":0.65,"gamma":0.05,"t_0":10,
             "kappa":0.75,"max_second_per_sample":0.1,"obj_fun":"accept_rate","par_type":default_par_type(name)}

    return(out)
#def opt_default_arguments(name):

dual_arguments = [ep_dual_metadata_argument,evolve_L_opt_metadata_argument,alpha_opt_metadata_argument]
opt_arguments = [gpyopt_fast_metatdata_argument,gpyopt_medium_metadata_argument,gpyopt_slow_metadata_argument]
# shared by all chains in the same sampling session
fast_tune_setting_dict = {"epsilon":ep_dual_metadata_argument}
medium_tune_setting_dict = {"gpyopt":gpyopt_medium_metadata_argument}
slow_tune_setting_dict = {"gpyopt":gpyopt_slow_metadata_argument}

tune_settings_dict = {"fast":fast_tune_setting_dict,"medium":medium_tune_setting_dict,"slow":slow_tune_setting_dict}

# controls every tuning paramter for an mcmc_sampler object
def tuning_settings(dual_arguments,opt_arguments):

    out = {}
    fast_tune_setting_dict = {"dual":[],"opt":[]}
    medium_tune_setting_dict = {"dual":[],"opt":[],"adapt":[]}
    slow_tune_setting_dict = {"dual":[],"opt":[],"adapt":[]}

    for obj in dual_arguments:
        if obj["par_type"]=="fast":
            fast_tune_setting_dict["dual"].append(obj)
        if obj["par_type"]=="medium":
            medium_tune_setting_dict["dual"].append(obj)
        if obj["par_type"]=="slow":
            slow_tune_setting_dict["dual"].append(obj)

    for obj in opt_arguments:
        if obj["par_type"]=="fast":
            fast_tune_setting_dict["opt"].append(obj)
        if obj["par_type"]=="medium":
            medium_tune_setting_dict["opt"].append(obj)
        if obj["par_type"]=="slow":
            slow_tune_setting_dict["opt"].append(obj)

    # at this point should look at all tuning parameters and fill in by default values any yet to be specified variables




    for param,val in tune_dict:
        if param.par_type=="fast":
            if val == "dual":

                tlist = fast_tune_setting_dict["dual"]
                exists = False
                for arg in tlist:
                    if param.name in arg:
                        exists = True
                tlist.append(dual_default_arguments(name=param.name))

            elif val=="opt":
                param_names = slow_dict and opt_dict
                chosen_list = {}
                for param in param_names:
                    chosen_list.update({param,False})
                tlist = fast_tune_setting_dict["opt"]
                for arg in tlist:
                    if param.name in arg["params"]:
                        chosen_list.update({param,True})


        if param.par_type=="medium":
            if val == "dual":
                tlist = medium_tune_setting_dict["dual"]
            elif val=="opt":
                tlist = medium_tune_setting_dict["opt"]
            elif val=="adapt"
                tlist = medium_tune_setting_dict["adapt"]
        if param.par_type=="slow":
            if val == "dual":
                tlist = slow_tune_setting_dict["dual"]
            elif val=="opt":
                tlist = slow_tune_setting_dict["opt"]
            elif val=="adapt":
                tlist = slow_tune_setting_dict["adapt"]


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