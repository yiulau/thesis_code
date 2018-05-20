from adapt_util.adapt_util import tuneable_param,welford_tensor
from adapt_util.return_update_list import return_update_lists
from adapt_util.tune_param_classes.tune_param_class import tune_param_creator
import GPyOpt,numpy,torch,math
from adapt_util.tune_param_classes.tuning_param_obj import tuning_param_states
from adapt_util.tune_param_classes.tune_param_setting_util import default_adapter_setting
class adapter_class(object):
    #
    def __init__(self,one_chain_obj,adapter_setting=None):

        self.one_chain_experiment = one_chain_obj
        #print(one_chain_obj.__dict__.keys())
        #print(one_chain_obj.sampling_metadata.__dict__)
        #exit()
        # fast = tuned by dual. update every iter in ini_buffer and end_buffer
        # medium = tuned by dual or opt. update every window_size
        # slow = tuned by dual,adapt or opt, update every cur_window_size, which doubles after each update
        # default definitions
        # par_type should be determined by setting default parameters
        self.par_type_dict = {}
        for param,obj in self.one_chain_experiment.tune_settings_dict["par_name"].items():
            self.par_type_dict.update({param:obj["par_type"]})


        self.permitted_tune_params = ("epsilon","evolve_L","evolve_t","alpha","xhmc_delta","cov")
        #self.par_type_dict = {"epsilon":"fast","evolve_L": "medium", "evolve_t": "medium", "alpha": "medium", "xhmc_delta": "medium",
        #                 "diag_cov": "slow", "cov": "slow"}
        #print(one_chain_obj.tune_dict.keys())
        #print(one_chain_obj.tune_dict)
        #exit()
        #self.tuneable_param = {}
        # tuneable_param stores tuning parameters'
        #for param,val in one_chain_obj.tune_dict.items():
        #    if param in self.par_type_dict:
        #        self.tuneable_param.update({param:val})
        #print(self.tuneable_param)
        self.dynamic = self.one_chain_experiment.tune_dict["dynamic"]
        self.second_order = self.one_chain_experiment.tune_dict["second_order"]
        self.metric_name = self.one_chain_experiment.tune_dict["metric_name"]
        if self.dynamic:
            self.criterion = self.one_chain_experiment.tune_dict["criterion"]
        else:
            self.criterion = None
            self.one_chain_experiment.tune_dict.update({"criterion":None})
        #self.tuneable_params_name = tuneable_param(self.dynamic,self.second_order,self.metric_name,self.criterion,
        #                                           )
        # unique for individual sampler
        if self.dynamic==False:
            if not self.criterion is None:
                raise ValueError("static integrator should not have termination criterion. put None instead")
        #self.tuning_obj =
        if adapter_setting is None:
            self.adapter_setting = default_adapter_setting()
        else:
            self.adapter_setting = adapter_setting
        #self.adapter_meta = adapter_settings(one_chain_obj.sampling_metadata)
        #ep_obj = self.param_obj_dict["epsilon"]
        #if self.ep_obj.tune_method=="opt":
        #    self.par_type_dict.update({"epsilon":"medium"})

        self.tune_method_dict = {}
        for param,val in one_chain_obj.tune_dict.items():
            #print(val)
            if param in self.permitted_tune_params:
                if param=="cov":
                    if val == "adapt":
                        self.tune_method_dict.update({param:val})
                    else:
                        self.tune_method_dict.update({param: "fixed"})
                        self.par_type_dict.update({param: "fixed"})
                else:
                    if not val=="opt" and not val=="dual" and not val is None:
                        self.tune_method_dict.update({param:"fixed"})
                        self.par_type_dict.update({param:"fixed"})
                    elif val=="opt" or val=="dual":
                        self.tune_method_dict.update({param:val})

        #print(self.tune_method_dict)
        #if self.tune_method_dict["epsilon"]=="opt":
        #    print(self.par_type_dict)
        #    exit()
        #    raise ValueError("fix this")

        #    self.par_type_dict.update({"epsilon":"medium"})

        #print(self.par_type_dict)

        self.fast_dict = {}
        self.medium_dict = {}
        self.slow_dict = {}
        self.dual_dict = {}
        self.opt_dict = {}
        self.adapt_dict = {}
        self.tune_fast = False
        self.tune_medium = False
        self.tune_slow = False
        for param,val in one_chain_obj.tune_dict.items():
            if param in self.par_type_dict:
                # tune_method one of {fixed,dual,opt,adapt}
                tune_method = self.tune_method_dict[param]
                # par_type one of {fast,medium,slow}
                par_type = self.par_type_dict[param]
                if tune_method == "fixed":
                    pass
                elif tune_method == "dual":
                    self.dual_dict.update({param:par_type})
                elif tune_method == "opt":
                    self.opt_dict.update({param:par_type})
                elif tune_method == "adapt":
                    self.adapt_dict.update({param:par_type})
                else:
                    raise ValueError("unknown tune method")
                if par_type == "fast":
                    self.fast_dict.update({param:tune_method})
                    self.tune_fast = True
                elif par_type == "medium":
                    self.medium_dict.update({param:tune_method})
                    self.tune_medium = True
                elif par_type == "slow":
                    self.slow_dict.update({param:tune_method})
                    self.tune_slow = True
                elif par_type == "fixed":
                    pass
                else:
                    raise ValueError("unknow par type")

                #assert self.par_type_dict[param] == self.one_chain_experiment.tune_settings_dict["par_name"][param]["par_type"]

        self.adapter_meta = adapter_metadata(self.one_chain_experiment.chain_setting,
                                             self.tune_fast, self.tune_medium, self.tune_slow)

        #self.update_fast_list, self.update_medium_list, self.update_slow_list = return_update_lists(self.adapter_meta)
        if self.adapter_meta.tune:
            #print("yes_tune")

            self.choose_iter_dict = return_update_lists(self.adapter_meta,self.adapter_setting)
            self.update_fast_list = self.choose_iter_dict["fast"]
            self.update_medium_list = self.choose_iter_dict["medium"]
            self.update_slow_list = self.choose_iter_dict["slow"]
            #print(self.choose_iter_dict)
            #exit()







    # for each
    # first determine if there are fast parameters
    def update_list(self):
        pass

    def update(self,sample_obj):
        # by definition
        assert hasattr(self,"tuning_param_states")
        slow_state = self.tuning_param_states.slow_state
        medium_state = self.tuning_param_states.medium_state
        fast_state = self.tuning_param_states.fast_state

        if not fast_state is None:
            if hasattr(fast_state,"dual_state"):
                fast_state.dual_state.update(sample_obj)
            if hasattr(fast_state,"opt_state"):
                fast_state.opt_state.update(sample_obj)
            if hasattr(fast_state,"adapt_cov_state"):
                fast_state.adapt_cov_state.update(sample_obj)

        if not medium_state is None:
            if hasattr(medium_state,"dual_state"):
                medium_state.dual_state.update(sample_obj)
            if hasattr(medium_state,"opt_state"):
                medium_state.opt_state.update(sample_obj)
            if hasattr(medium_state,"adapt_cov_state"):
                medium_state.adapt_cov_state.update(sample_obj)

        if not slow_state is None:
            if hasattr(slow_state,"dual_state"):
                slow_state.dual_state.update(sample_obj)
            if hasattr(slow_state,"opt_state"):
                slow_state.opt_state.update(sample_obj)
            if hasattr(slow_state,"adapt_cov_state"):
                slow_state.adapt_cov_state.update(sample_obj)


    #def prepare_adapter(self):
    #    self.tuning_param_state = tuning_param_states(self)





#class dual_settings(object):
#    def __init__(self,ini_buffer=75,end_buffer=50):
#        self.ini_buffer = ini_buffer
#        self.end_buffer = end_buffer



#class opt_settings(object):
#    def __init__(self,min_medium_updates=10,):
#        self.min_medium_updates = min_medium_updates

#class adapt_settings(object):
#    def __init__(self,min_slow_updates):
#        self.min_slow_updates = min_slow_updates
class adapter_metadata(object):
    # records metadata about the adapter for a sampler obj
    # does not set anything
    def __init__(self,chain_setting,tune_fast,tune_medium,tune_slow):
        self.num_samples = chain_setting["num_samples"]
        self.tune_l = chain_setting["tune_l"]
        self.tune_fast = tune_fast
        self.tune_medium = tune_medium
        self.tune_slow = tune_slow
        self.tune = self.tune_fast or self.tune_medium or self.tune_slow


def adapter_metadata_dict(chain_setting,tune_fast,tune_medium,tune_slow):
    # records metadata about the adapter for a sampler obj
    # does not set anything
    #def __init__(self,sampling_metadata,tune_fast,tune_medium,tune_slow):
    out = {"num_samples":chain_setting.num_samples,"tune_l":chain_setting.tune_l,"tune_fast":tune_fast}
    out.update({"tune_medium":tune_medium,"tune_slow":tune_slow})
    tune = tune_fast or tune_medium or tune_slow
    out.update({"tune":tune})
    return(out)





#class tuning_param_settings(object):
  #  def __init__(self):
