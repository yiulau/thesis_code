import abc, numpy, pickle, os
from abstract.mcmc_sampler import mcmc_sampler_settings,mcmc_sampler

class experiment(object):
    #__metaclass__ = abc.ABCMeta

    def __init__(self,input_object=None,experiment_metaobj=None):


        self.input_object = input_object
        self.tune_param_grid = self.input_object.tune_param_grid
        self.store_grid_obj = numpy.empty(self.input_object.grid_shape,dtype=object)
        #loop through each point in the grid and initiate an sampling_object
        it = numpy.nditer(self.store_grid_obj, flags=['multi_index',"refs_ok"])
        cur = 0
        self.id_to_multi_index = []
        self.multi_index_to_id = {}
        while not it.finished:

            self.id_to_multi_index.append(it.multi_index)
            self.multi_index_to_id.update({it.multi_index: cur})
            tune_dict = self.tune_param_grid[it.multi_index]
            sampling_metaobj = mcmc_sampler_settings(mcmc_id = cur)
            grid_pt_metadict = {"mcmc_id":cur}
            self.store_grid_obj[it.multi_index] = {"sampler":mcmc_sampler(tune_dict,sampling_metaobj),"metadata":grid_pt_metadict}
            it.iternext()
            cur +=1



    def pre_experiment_diagnostics(self,test_run_chain_length=15):

    # 1 estimated total volume
    # 2 estimated  computing time (per chain )
    # 3 estimated total computing time (serial)
    # 4 estimated total coputing time (if parallel, given number of agents)
    # 5 ave number of time per leapfrog

        experiment_meta_obj = experiment_meta(chain_length=test_run_chain_length)
        temp_experiment = self.experiment.clone(experiment_meta_obj)

        temp_output = temp_experiment.run()
        out = {}
        time = temp_output.sampling_metadata.total_time
        ave_per_leapfrog = temp_output.sample_metadata.ave_second_per_leapfrog
        total_size = temp_output.sample_metadata.size_mb
        estimated_total_volume = total_size * (self.num_chains * self.num_per_chain) / test_run_chain_length
        out.append({"total_volume": estimated_total_volume})
        estimated_compute_seconds_per_chain = time * self.num_per_chain / test_run_chain_length
        out.append({"seconds_per_chain": estimated_compute_seconds_per_chain})
        estimated_compute_seconds = self.num_chains * estimated_compute_seconds_per_chain
        out.append({"total_seconds": estimated_compute_seconds})
        estimated_compute_seconds_parallel = estimated_compute_seconds / self.num_agents
        out.append({"total_seconds with parallel": estimated_compute_seconds_parallel})
        with open('model.pkl', 'wb') as f:
            pickle.dump(temp_output, f)
        size = os.path.getsize("./model.pkl") / (1024. * 1024)
        os.remove("./model.pkl")


    def run(self):
        it = numpy.nditer(self.store_grid_obj, flags=['multi_index', "refs_ok"])
        while not it.finished:
            sampler = self.store_grid_obj[it.multi_index]["sampler"]
            result = sampler.start_sampling()
            self.store_grid_obj[it.multi_index].update({"result":result})
            it.iternext()
        return()

    def clone(self):
        out = experiment(self.experiment_metaobj,self.input_object.clone())




#input_dict = {"v_fun":[1,2],"alpha":[0],"epsilon":[0.1,0.2,0.3],"second_order":[True,False]}

#input_obj = tuneinput_class()

#exper_obj = experiment(input_object=input_object)

#print(exper_obj.grid_shape)
#print(exper_obj.param_name_list)
#exper_obj2 = exper_obj.clone()
#exit()
#print(exper_obj.ep)

#print(exper_obj.__dict__)
#exit()
#print(experiment.__dict__)

class experiment_meta(object):
    def __init__(self,chain_length):
        self.chain_length = chain_length



#class experiment_meta(object):
    #def __init__(self):


class tuneinput_class(object):
    permissible_var_names = ("v_fun", "dynamic", "windowed", "second_order", "criterion", "metric_name", "epsilon", "evolve_t",
                             "evolve_L", "alpha", "xhmc_delta", "Cov")

    permissible_var_values = {"dynamic": (True, False)}
    permissible_var_values.update({"windowed":(True,False)})
    permissible_var_values.update({"second_order": (True, False)})
    permissible_var_values.update({"criterion": ("nuts", "gnuts", "xhmc",None)})
    permissible_var_values.update({"metric_name": ("unit_e", "diag_e", "dense_e", "softabs_e",
                                              "softabs_diag", "softabs_op", "softabs_op_diag")})
    permissible_var_values.update({"epsilon": ("dual", "opt")})
    permissible_var_values.update({"evolve_t": ("dual", "opt", None)})
    permissible_var_values.update({"evolve_L": ("dual", "opt", None)})
    permissible_var_values.update({"alpha": ("dual", "opt", None)})
    permissible_var_values.update({"xhmc_delta": ("dual", "opt", None)})
    permissible_var_values.update({"cov": ("adapt", None)})

    def __init__(self,input_dict=None):
        self.input_dict = input_dict
        self.grid_shape = []
        self.param_name_list = []
        for param_name,val_list in input_dict.items():
            if not param_name in self.permissible_var_names:
                print(param_name)
                raise ValueError("not one of permissible attributes")
            elif len(val_list)==0:
                raise ValueError("can't have empty list ")
            else:
                if param_name=="v_fun":
                    pass
                else:
                    for option in val_list:
                        pass
                        #if not "fixed" in self.permissible_var_values[param_name]:
                            #if not option in self.permissible_var_values[param_name]:
                              #  print(param_name)
                              #  print(option)
                              #  raise ValueError("not one of permitted options for this attribute")
            setattr(self,param_name,val_list)
            #self.grid_shape.append(len(val_list))
            #self.param_name_list.append(param_name)

        for name in self.permissible_var_names:
            if hasattr(self,name):
                self.grid_shape.append(len(getattr(self,name)))
                self.param_name_list.append(name)

        self.tune_param_grid = numpy.empty(self.grid_shape)



        # store tuning parameters value in a grid
        self.tune_param_grid = numpy.empty(self.grid_shape,dtype=object)
        #print(self.tune_param_grid[0,0,0,0])
        it = numpy.nditer(self.tune_param_grid,flags=["multi_index","refs_ok"])
        while not it.finished:
            settings_dict = {}
            for i in range(len(self.param_name_list)):
                settings_dict.update({self.param_name_list[i]:getattr(self,self.param_name_list[i])[it.multi_index[i]]})
            self.tune_param_grid[it.multi_index] = settings_dict
            #print(it.multi_index)
            it.iternext()

    def singleton_tune_dict(self):
        if self.tune_param_grid.size!=1:
            raise ValueError("not a singleton")
        else:
            it = numpy.nditer(self.tune_param_grid, flags=["multi_index", "refs_ok"])
            tune_dict = self.tune_param_grid[it.multi_index]
        return(tune_dict)



    def clone(self):
        if hasattr(self,"Cov"):
            if not getattr(self,"Cov")=="adapt":
                copy = getattr(self,"Cov")[0].clone()

        input_dict = self.input_dict.copy()
        #print(input_dict)

        input_dict["Cov"] = [copy]
        out = tuneinput_class(input_dict)
        return(out)


