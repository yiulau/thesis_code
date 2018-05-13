import abc,math, torch
from adapt_util.find_reasonable_start import find_reasonable_ep
from adapt_util.tune_param_classes.tuning_param_obj import dual_param_metadata

class tune_param_concrete(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,update_iter_list,tune_method,par_type,name,tune_setting=None):

        # with its name as key
        self.update_iter_list = update_iter_list
        self.tune_method = tune_method
        self.par_type = par_type
        self.name = name
        self.next_refresh_iter = self.update_iter_list[1]
        self.cur_in_iter_list = 0
        self.tune_setting = tune_setting

    def update_internal_state(self,sample_obj):
        if self.tune_method=="dual":
            input = self.obj_fun(sample_obj)
            self.dual_metadata.update_dual(input)
        elif self.tune_method=="opt":
            self.gpyopt_obj.update(sample_obj)
        elif self.tune_method == "adapt":
            self.adapt_obj.update(sample_obj)
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method ")

    def initialize_tuning_param(self):
        if self.need_initialize:
            init = self.find_reasonable_start()
            self.set_val(init)
            self.need_initialize = False
        return()


    @abc.abstractmethod
    def find_reasonable_start(self):
        # returns reasonable initialization value
        return ()
    @abc.abstractmethod
    def find_bounds(self):
        #return
        return()


def dual_input_accept_rate(sample_obj,dual_settings_obj):
    alpha = sample_obj.accept_rate
    input = dual_settings_obj.target - alpha
    return(input)



class epsilon(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type):
        super(epsilon, self).__init__(update_iter_list,tune_method,par_type)
        self.update_priority = 1
        self.store = torch.zeros(1)
        self.cur_val = None
        self.name = "epsilon"
        if tune_method=="dual":
            if self.tune_setting is None:
                self.dual_metadata = dual_param_metadata(self.name)
            else
                self.dual_metadata = dual_param_metadata(**self.tune_setting)
        #elif tune_method=="opt":
        #    if self.tune_setting is None:
        #        self.
         #   self.settings = opt_settings(self.name)
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        epsilon = find_reasonable_ep(self.tuning_obj)
        return(epsilon)

    def find_bounds(self):

        if self.tune_method=="fixed":
            self.bounds = (self.cur_val*0.9,self.cur_val*1.1)
        else:
            self.bounds = (1e-6, 0.5)
        return(self.bounds)

    def get_val(self):
        return(self.store[0])

    def set_val(self,val):
        self.store[0]=val
        self.cur_val = self.store[0]
        return()

class evolve_t(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type):
        super(evolve_t, self).__init__(update_iter_list,tune_method,par_type)
        self.update_priority = 2
        self.store = torch.zeros(1)
        self.cur_val = None
        self.ave_second_per_leapfrog = self.integrator.find_ave_second_per_leapfrog()
        self.max_second_per_sample = self.settings.max_second_per_sample
        self.name="evolve_t"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(self.name)
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        out = (self.bounds[0] + self.bounds[1])*0.5
        return(epsilon)

    def find_bounds(self):
        ep_bounds = self.tuning_object.param_dict["ep"].find_bounds()
        max_t = (self.maximum_second_per_sample / self.ave_second_per_leapfrog) * ep_bounds[1]
        min_t = ep_bounds[1] * 3.1
        self.bounds = (min_t,max_t)
        return(ep_bounds)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class evolve_L(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type,param_obj_dict):
        super(evolve_L, self).__init__(update_iter_list,tune_method,par_type)
        self.update_priority = 2
        self.store = torch.zeros(1)
        self.cur_val = None
        self.name = "evolve_L"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(self.name)
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        out = round((self.bounds[0]+self.bounds[1])*0.5)
        return(epsilon)

    def find_bounds(self):
        ep_bounds = self.tuning_object.param_dict["ep"].find_bounds()
        max_L = round((self.maximum_second_per_sample / self.ave_second_per_leapfrog) + 1 )
        min_L = 1
        self.bounds = (min_L, max_L)
        return(self.bounds)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class alpha(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type,tuning_obj):
        super(alpha, self).__init__(update_iter_list,tune_method,par_type)
        self.update_priority = 3
        self.store = torch.zeros(1)
        self.cur_val = None
        self.name = "alpha"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(self.name)


    def find_reasonable_start(self):
        # unit_e unit trajectory length
        reasonable_start = 1
        return(reasonable_start)

    def find_bounds(self):
        self.bounds = (1e-6, 1e6)
        return(self.bounds)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class xhmc_delta(tune_param_concrete):
    def __init__(self, update_iter_list, tune_method, par_type,tuning_obj):
        super(xhmc_delta, self).__init__(update_iter_list, tune_method,par_type, tuning_obj)
        self.update_priority = 4
        self.store = torch.zeros(1)
        self.cur_val = None
        self.name="xhmc_delta"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(self.name)

    def find_reasonable_start(self):
        # unit_e unit trajectory length
        reasonable_start = 0.1
        return (reasonable_start)

    def find_bounds(self):
        out = (1e-6, 0.2)
        self.bounds= out
        return(out)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class dense_cov(tune_param_concrete):
    def __init__(self, update_iter_list, tune_method, par_type, tuning_obj):
        super(dense_cov, self).__init__(update_iter_list, tune_method, par_type, tuning_obj)
        self.update_priority = 5
        self.store = torch.zeros(tuning_obj.dim,tuning_obj.dim)
        self.cur_val = None
        self.name = "dense_cov"

        if self.tune_method == "adapt":
            self.adapt_obj = self.tuning_obj.diagnostics_obj.adapt_obj
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method")
    def find_reasonable_start(self):
        # unit matrix
        pass

    def find_bounds(self):
        pass

    def get_val(self):
        return (self.store.clone())

    def set_val(self, val):
        self.store.copy_(val)
        self.cur_val = self.store.clone()
        return ()

class diag_cov(tune_param_concrete):
    def __init__(self, update_iter_list, tune_method, par_type, tuning_obj):
        super(diag_cov, self).__init__(update_iter_list, tune_method, par_type, tuning_obj)
        self.update_priority = 5
        self.store = torch.zeros(tuning_obj.dim)
        self.cur_val = None
        self.name="diag_cov"
        if self.tune_method == "adapt":
            self.adapt_obj = self.tuning_obj.diagnostics_obj.adapt_obj
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method")

    def find_reasonable_start(self):
        # unit matrix
        pass

    def find_bounds(self):
        pass

    def get_val(self):
        return (self.store.clone())

    def set_val(self, val):
        self.store.copy_(val)
        self.cur_val = self.store.clone()
        return()


def tune_param_creator(param_name,iter_list=None,tune_method=None,par_type=None,par_tune_setting=None):
    if param_name=="epsilon":
        out = epsilon(iter_list,tune_method,par_type,par_tune_setting)
    if param_name=="evolve_L":
        out = evolve_L(iter_list,tune_method,par_type,par_tune_setting)
    if param_name=="evolve_t":
        out = evolve_t(iter_list,tune_method,par_type,par_tune_setting  )
    if param_name=="alpha":
        out = alpha(iter_list, tune_method, par_type,par_tune_setting)
    if param_name=="xhmc_delta":
        out = xhmc_delta(iter_list, tune_method, par_type,par_tune_setting)
    if param_name=="dense_cov":
        out = dense_cov(iter_list, tune_method, par_type,par_tune_setting)
    if param_name=="diag_cov":
        out = diag_cov(iter_list, tune_method, par_type,par_tune_setting)

    return(out)

def tune_params_obj_creator(tune_dict,adapter_obj,tune_settings_dict={}):
    # given tune_dict : describe what it contains (comes from input_class_objects)
    # contains key names like
    permitted_par_names = ("epsilon", "evolve_L", "evolve_t", "alpha", "xhmc_delta", "cov", "metric_name")
    params_obj_dict = {}
    activate_cov = False
    tuning_param_states = adapter_obj.tuning_param_states
    for param_name, val in tune_dict.items():
        if param_name=="metric_name":
            if val == "dense_e" or val == "diag_e":
                if val=="dense_e":
                    cov_type = "dense"
                if val=="diag_e":
                    cov_type = "diag"

        elif param_name=="cov":
            activate_cov = True
        # only these params reach this point ("epsilon", "evolve_L", "evolve_t", "alpha", "xhmc_delta")
        # val could be float/double or adapt,opt
        elif param_name in permitted_par_names:
            tune_method = adapter_obj.tune_method_dict[param_name]
            par_type = adapter_obj.par_type_dict[param_name]
            iter_list = adapter_obj.choose_iter_list(par_type)
            if param_name in tune_settings_dict:
                par_tune_setting = tune_settings_dict[param_name]
            else:
                par_tune_setting = None
            obj = tune_param_creator(param_name,iter_list,tune_method,par_type,par_tune_setting)
            if tune_method == "opt" or tune_method =="dual":
                #obj.initialize_tuning_param()
                obj.need_initialize = True
            else:
                obj.need_initialize = False
                obj.set_val(val)
            params_obj_dict.update({param_name:obj})

    if activate_cov:
        cov_tensor = tune_dict["cov"]
        if cov_type=="dense":
            obj = tune_param_creator("dense_cov")

        else:
            obj = tune_param_creator("diag_cov")

        obj.set_val(cov_tensor)
        params_obj_dict.update({"cov": obj})
    return(params_obj_dict)










