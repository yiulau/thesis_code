from adapt_util.adapt_util import tuneable_param,welford_tensor
from adapt_util.return_update_list import return_update_lists
from adapt_util.tune_param_classes.tune_param_class import tune_param_creator
import GPyOpt,numpy,torch,math
from adapt_util.tune_param_classes.tuning_param_obj import tuning_param_states
class adapter_class(object):
    #
    def __init__(self,one_chain_obj):

        self.one_chain_experiment = one_chain_obj
        #print(one_chain_obj.__dict__.keys())
        #print(one_chain_obj.sampling_metadata.__dict__)
        #exit()
        # fast = tuned by dual. update every iter in ini_buffer and end_buffer
        # medium = tuned by dual or opt. update every window_size
        # slow = tuned by dual,adapt or opt, update every cur_window_size, which doubles after each update
        # default definitions
        self.par_type_dict = {"epsilon":"fast","evolve_L": "medium", "evolve_t": "medium", "alpha": "medium", "xhmc_delta": "medium",
                         "diag_cov": "slow", "cov": "slow"}
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
        self.criterion = self.one_chain_experiment.tune_dict["criterion"]
        #self.tuneable_params_name = tuneable_param(self.dynamic,self.second_order,self.metric_name,self.criterion,
        #                                           )
        # unique for individual sampler
        if self.dynamic==False:
            if not self.criterion is None:
                raise ValueError("static integrator should not have termination criterion. put None instead")
        #self.tuning_obj =

        #self.adapter_meta = adapter_settings(one_chain_obj.sampling_metadata)
        #ep_obj = self.param_obj_dict["epsilon"]
        #if self.ep_obj.tune_method=="opt":
        #    self.par_type_dict.update({"epsilon":"medium"})

        self.tune_method_dict = {}
        for param,val in one_chain_obj.tune_dict.items():
            #print(val)
            if not val=="opt" and not val=="dual" and not val is None:
                self.tune_method_dict.update({param:"fixed"})
            elif val=="opt" or val=="dual":
                self.tune_method_dict.update({param:val})

        #print(self.tune_method_dict)
        if self.tune_method_dict["epsilon"]=="opt":
            self.par_type_dict.update({"epsilon":"medium"})

        #print(self.par_type_dict)

        self.fast_dict = {}
        self.medium_dict = {}
        self.slow_dict = {}
        self.dual_dict = {}
        self.opt_dict = {}
        self.adpat_dict = {}
        self.tune_fast = False
        self.tune_medium = False
        self.tune_slow = False
        for param,val in one_chain_obj.tune_dict:
            # tune_method one of {fixed,dual,opt,adapt}
            tune_method = self.tune_method_dict[param]
            # par_type one of {fast,medium,slow}
            par_type = self.par_type_dict[param]
            if tune_method == "fixed":
                pass
            elif tune_method == "dual":
                self.dual_dict.update({param:par_type})
            elif tune_method == "opt":
                self.opt_dict({param:par_type})
            elif tune_method == "adapt":
                self.adapt_dict({param:par_type})
            else:
                raise ValueError("unknown tune method")
            if par_type == "fast":
                self.fast_dict({param:tune_method})
                self.tune_fast = True
            elif par_type == "medium":
                self.medium_dict({param:tune_method})
                self.tune_medium = True
            elif par_type == "slow":
                self.slow_dict({param:tune_method})
                self.tune_slow = True
            else:
                raise ValueError("unknow par type")

        self.adapter_meta = adapter_settings(self.one_chain_experiment.sampling_metadata,
                                             self.tune_fast, self.tune_medium, self.tune_slow)

        #self.update_fast_list, self.update_medium_list, self.update_slow_list = return_update_lists(self.adapter_meta)
        choose_iter_dict = return_update_lists(self.adapter_meta)
        self.update_fast_list = choose_iter_dict["fast"]
        self.update_medium_list = choose_iter_dict["medium"]
        self.update_fast_list = choose_iter_dict["slow"]





    # for each
    # first determine if there are fast parameters
    def update_list(self):
        pass

    def update(self,sample_obj):
        # by definition
        slow_state = self.tuning_param_state.slow_state
        medium_state = self.tuning_param_state.medium_state
        fast_state = self.tuning_param_state.fast_state

        if not fast_state is None:
            if hasattr(fast_state,"dual_state"):
                fast_state.dual_state.update(sample_obj)

        if not medium_state is None:
            if hasattr(medium_state,"dual_state"):
                medium_state.dual_state.update(sample_obj)
            if hasattr(medium_state,"opt_state"):
                medium_state.opt_state.update(sample_obj)
            if hasattr(medium_state,"adapt_state"):
                medium_state.adapt_state.update(sample_obj)

        if not slow_state is None:
            if hasattr(slow_state,"dual_state"):
                slow_state.dual_state.update(sample_obj)
            if hasattr(slow_state,"opt_state"):
                slow_state.opt_state.update(sample_obj)
            if hasattr(slow_state,"adapt_state"):
                slow_state.adapt_state.update(sample_obj)


    def prepare_adapter(self):
        self.tuning_param_state = tuning_param_states(self)





class dual_settings(object):
    def __init__(self,ini_buffer=75,end_buffer=50):
        self.ini_buffer = ini_buffer
        self.end_buffer = end_buffer



class opt_settings(object):
    def __init__(self,min_medium_updates=10,):
        self.min_medium_updates = min_medium_updates

class adapt_settings(object):
    def __init__(self,min_slow_updates):
        self.min_slow_updates = min_slow_updates
class adapter_settings(object):
    def __init__(self,sampling_metadata,tune_fast,tune_medium,tune_slow,window_size=25):
        self.num_samples = sampling_metadata.num_samples
        self.tune_l = sampling_metadata.tune_l
        self.window_size = window_size
        self.tune_fast = tune_fast
        self.tune_medium = tune_medium
        self.tune_slow = tune_slow





#class tuning_param_settings(object):
  #  def __init__(self):
