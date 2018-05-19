import abc,math, torch
from adapt_util.find_reasonable_start import find_reasonable_ep
from adapt_util.tune_param_classes.tuning_param_obj import dual_param_metadata

class tune_param_concrete(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,update_iter_list,tune_method,par_type,name,par_tune_setting):

        # with its name as key
        self.update_iter_list = update_iter_list
        self.tune_method = tune_method
        self.par_type = par_type
        self.name = name
        if not self.par_type=="fixed":
            print(self.name)
            print(self.par_type)
            self.next_refresh_iter = self.update_iter_list[1]
            self.cur_in_iter_list = 0
        else:
            self.next_refresh_iter = None
            self.cur_in_iter_list = None
        if not par_tune_setting is None:
            self.maximum_second_per_sample = par_tune_setting.get("maximum_second_per_sample",None)
        #print(self.maximum_second_per_sample)
        #exit()
            if self.maximum_second_per_sample is None:
                self.par_tune_setting = par_tune_setting
                raise ValueError("should not happen")
            else:
                par_tune_setting.pop("maximum_second_per_sample")
                self.par_tune_setting = par_tune_setting
        if self.tune_method == "opt" or self.tune_method == "dual":
            # obj.initialize_tuning_param()
            self.need_initialize = True
        else:
            self.need_initialize = False
    # def update_internal_state(self,sample_obj):
    #     if self.tune_method=="dual":
    #         input = self.obj_fun(sample_obj)
    #         self.dual_metadata.update_dual(input)
    #     elif self.tune_method=="opt":
    #         self.gpyopt_obj.update(sample_obj)
    #     elif self.tune_method == "adapt":
    #         self.adapt_obj.update(sample_obj)
    #     elif self.tune_method=="fixed":
    #         pass
    #     else:
    #         raise ValueError("unknown tune method ")
    def set_Ham(self,Ham):
        self.Ham = Ham

    def initialize_tuning_param(self):
        #print("yes")
        #print(self.need_initialize)
        #exit()
        if self.need_initialize:
            init = self.find_reasonable_start()
            self.set_val(init)
            self.need_initialize = False
        return()

    # apply to each param obj after all have been created
    def add_param_objs_dict(self,param_objs_dict):
        assert self == param_objs_dict[self.name]
        self.param_objs_dict = param_objs_dict
    @abc.abstractmethod
    def find_reasonable_start(self):
        # returns reasonable initialization value
        return ()
    @abc.abstractmethod
    def find_bounds(self):
        #return
        return()

    @abc.abstractmethod
    def set_val(self,val):
        return

    @abc.abstractmethod
    def get_val(self):
        return
def dual_input_accept_rate(sample_obj,dual_settings_obj):
    alpha = sample_obj.accept_rate
    input = dual_settings_obj.target - alpha
    return(input)



class epsilon(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type,par_tune_setting):
        super(epsilon, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,par_type=par_type,
                                      name="epsilon",par_tune_setting=par_tune_setting)
        self.update_priority = 1
        self.store = torch.zeros(1)
        self.cur_val = None
        self.default_bounds = (1e-6, 0.5)
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(setting=self.par_tune_setting)
        elif tune_method=="opt":
            if "bounds" in par_tune_setting:
                self.default_bounds = par_tune_setting["bounds"]


    def find_reasonable_start(self):
        # unit_e unit trajectory length
        assert hasattr(self,"Ham")
        epsilon = find_reasonable_ep(self.Ham)
        return(epsilon)

    def find_bounds(self):
        if self.tune_method=="fixed":
            self.bounds = (self.cur_val*0.9,self.cur_val*1.1)
        else:
            self.bounds = self.default_bounds
        return(self.bounds)

    def get_val(self):
        return(self.store[0])

    def set_val(self,val):
        self.store[0]=val
        self.cur_val = self.store[0]
        return()

class evolve_t(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type,par_tune_setting):
        super(evolve_t, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,par_type=par_type,
                                      name="evolve_t",par_tune_setting=par_tune_setting)
        self.update_priority = 2
        self.store = torch.zeros(1)
        self.cur_val = None
        self.default_bounds = None
        #self.ave_second_per_leapfrog = self.integrator.find_ave_second_per_leapfrog()
        #self.maximum_second_per_sample = self.par_tune_setting.max_second_per_sample
        self.name="evolve_t"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(setting=self.par_tune_setting)
        elif tune_method=="opt":
            if "bounds" in par_tune_setting:
                self.default_bounds = par_tune_setting["bounds"]
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        out = (self.bounds[0] + self.bounds[1])*0.5
        return(out)

    def find_bounds(self):
        if self.default_bounds is None:
            self.ave_second_per_leapfrog = self.integrator.find_ave_second_per_leapfrog()
            ep_bounds = self.param_objs_dict["epsilon"].find_bounds()
            max_t = (self.maximum_second_per_sample / self.ave_second_per_leapfrog) * ep_bounds[1]
            min_t = ep_bounds[1] * 3.1
            assert min_t < max_t
            self.default_bounds = (min_t,max_t)
        else:
            self.bounds = self.default_bounds
        return(self.bounds)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class evolve_L(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type,par_tune_setting):
        super(evolve_L, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,par_type=par_type,
                                      name="evolve_L",par_tune_setting=par_tune_setting)
        self.update_priority = 2
        # this should be int. but we store as float. make sure to convert when getting value
        self.store = torch.zeros(1)
        self.cur_val = None
        self.default_bounds = None
        self.name = "evolve_L"
        # self.ave_second_per_leapfrog = self.integrator.find_ave_second_per_leapfrog()
        #self.maximum_second_per_sample = self.par_tune_setting.maximum_second_per_sample
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(setting=self.par_tune_setting)
        elif tune_method=="opt":
            if "bounds" in par_tune_setting:
                self.default_bounds = par_tune_setting["bounds"]
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        #self.find_bounds()
        #out = round((self.bounds[0]+self.bounds[1])*0.5)
        out = 8
        return(out)

    def find_bounds(self):
        if self.default_bounds is None:
            self.ave_second_per_leapfrog = self.integrator.find_ave_second_per_leapfrog()
            ep_bounds = self.param_objs_dict["epsilon"].find_bounds()
            max_L = round((self.maximum_second_per_sample / self.ave_second_per_leapfrog) + 1 )
            min_L = 1
            assert min_L < max_L
            self.default_bounds = (min_L, max_L)
        else:
            self.bounds = self.default_bounds
        return(self.bounds)

    def get_val(self):

        return (int(self.store[0]))

    def set_val(self, val):
        val = int(val)
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class alpha(tune_param_concrete):
    def __init__(self,update_iter_list,tune_method,par_type,par_tune_setting):
        #super(alpha, self).__init__(update_iter_list,tune_method,par_type)
        super(alpha, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,par_type=par_type,
                                    name="alpha",par_tune_setting=par_tune_setting)
        self.update_priority = 3
        self.store = torch.zeros(1)
        self.cur_val = None
        self.default_bounds = (1e-6, 0.5)
        self.name = "alpha"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(setting=self.par_tune_setting)
        elif tune_method=="opt":
            if "bounds" in par_tune_setting:
                self.default_bounds = par_tune_setting["bounds"]

    def find_reasonable_start(self):
        # unit_e unit trajectory length
        reasonable_start = 1
        return(reasonable_start)

    def find_bounds(self):
        self.bounds = self.default_bounds

        return(self.bounds)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class xhmc_delta(tune_param_concrete):
    def __init__(self, update_iter_list, tune_method, par_type,tuning_obj,par_tune_setting):
        super(xhmc_delta, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,par_type=par_type,
                                    name="xhmc_delta",par_tune_setting=par_tune_setting)
        self.update_priority = 4
        self.store = torch.zeros(1)
        self.cur_val = None
        self.default_bounds = (1e-6, 0.2)
        self.name="xhmc_delta"
        if tune_method=="dual":
            self.dual_metadata = dual_param_metadata(setting=self.par_tune_setting)
        elif tune_method=="opt":
            if "bounds" in par_tune_setting:
                self.default_bounds = par_tune_setting["bounds"]
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        reasonable_start = 0.1
        return (reasonable_start)

    def find_bounds(self):

        self.bounds = self.default_bounds
        return(self.bounds)

    def get_val(self):
        return (self.store[0])

    def set_val(self, val):
        self.store[0] = val
        self.cur_val = self.store[0]
        return ()

class dense_cov(tune_param_concrete):
    def __init__(self, update_iter_list, tune_method, par_type,par_tune_setting):
        super(dense_cov, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,
                                        par_type=par_type,name="dense_cov",par_tune_setting=None)
        self.dim = par_tune_setting["dim"]
        self.update_priority = 5
        self.store = torch.zeros(self.dim,self.dim)
        self.cur_val = None
        self.name = "dense_cov"

        # if self.tune_method == "adapt":
        #     self.adapt_obj = self.tuning_obj.diagnostics_obj.adapt_obj
        # elif self.tune_method == "fixed":
        #     pass
        # else:
        #     raise ValueError("unknown tune method")
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
    def __init__(self, update_iter_list, tune_method, par_type,par_tune_setting):
        super(diag_cov, self).__init__(update_iter_list=update_iter_list,tune_method=tune_method,
                                        par_type=par_type,name="diag_cov",par_tune_setting=None)
        self.dim = par_tune_setting["dim"]
        self.update_priority = 5
        self.store = torch.zeros(self.dim)
        self.cur_val = None
        self.name="diag_cov"
        # if self.tune_method == "adapt":
        #     self.adapt_obj = self.tuning_obj.diagnostics_obj.adapt_obj
        # elif self.tune_method == "fixed":
        #     pass
        # else:
        #     raise ValueError("unknown tune method")

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

    def create_store(self,dim):
        self.store = torch.zeros(dim)

def tune_param_creator(param_name,iter_list,tune_method,par_type,par_tune_setting):
    if param_name=="epsilon":
        #out = epsilon(iter_list,tune_method,par_type,par_tune_setting)
        out = epsilon(update_iter_list=iter_list,tune_method=tune_method,
                      par_type=par_type,par_tune_setting=par_tune_setting)
    if param_name=="evolve_L":
        out = evolve_L(iter_list,tune_method,par_type,par_tune_setting)
    if param_name=="evolve_t":
        out = evolve_t(iter_list,tune_method,par_type,par_tune_setting)
    if param_name=="alpha":
        out = alpha(iter_list, tune_method, par_type,par_tune_setting)
    if param_name=="xhmc_delta":
        out = xhmc_delta(iter_list, tune_method, par_type,par_tune_setting)
    if param_name=="dense_cov":
        out = dense_cov(iter_list, tune_method, par_type,par_tune_setting)
    if param_name=="diag_cov":
        out = diag_cov(iter_list, tune_method, par_type,par_tune_setting)

    return(out)

def tune_param_objs_creator(tune_dict,adapter_obj,tune_settings_dict):
    # given tune_dict : describe what it contains (comes from input_class_objects)
    # contains key names like
    permitted_par_names = ("epsilon", "evolve_L", "evolve_t", "alpha", "xhmc_delta", "cov", "metric_name")
    params_obj_dict = {}
    activate_cov = False
    #tuning_param_states = adapter_obj.tuning_param_states
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
            if tune_method=="fixed":
                par_type= "fixed"
                assert adapter_obj.par_type_dict[param_name]==par_type
                iter_list=[]
            else:
                par_type = adapter_obj.par_type_dict[param_name]
                iter_list = adapter_obj.choose_iter_dict[par_type]
            #print(param_name)
            #print(par_type)
            #print(tune_settings_dict["par_type"]["fast"]["dual"])
            #exit()
            par_tune_setting = {}
            if tune_method=="dual":
                #par_tune_setting = tune_settings_dict["par_type"][par_type][tune_method][param_name]
                par_tune_setting.update(tune_settings_dict["par_name"][param_name])
                par_tune_setting.update(tune_settings_dict["others"])

            elif tune_method=="opt":
                #par_tune_setting = tune_settings_dict["par_type"][par_type][tune_method][param_name]
                print(tune_settings_dict["par_name"][param_name])
                par_tune_setting.update(tune_settings_dict["par_name"][param_name])
                par_tune_setting.update(tune_settings_dict["others"])
            elif tune_method=="fixed":
                par_tune_setting = None
            else:
                print(tune_method)
                #exit()
                #assert tune_method=="adapt_cov"

            #print(param_name)
            #print(par_tune_setting)
            print(param_name)
            print(iter_list)
            print(tune_method)
            print(par_type)
            print(par_tune_setting)

            obj = tune_param_creator(param_name=param_name,iter_list=iter_list,tune_method=tune_method,
                                     par_type=par_type,par_tune_setting=par_tune_setting)
            #obj = tune_param_creator(param_name=param_name,iter_list,tune_method,par_type,par_tune_setting)
            if not obj.need_initialize:
                obj.set_val(val)
            params_obj_dict.update({param_name:obj})

    if activate_cov:
        par_type = adapter_obj.par_type_dict["cov"]
        iter_list = adapter_obj.choose_iter_dict[par_type]
        cov_tensor = tune_dict["cov"]
        dim = cov_tensor.shape[0]
        if cov_type=="dense":
            obj = tune_param_creator(param_name="dense_cov",iter_list=iter_list,tune_method="adapt_cov",par_type=par_type,par_setting={"dim":dim})

        else:
            obj = tune_param_creator(param_name="diag_cov",iter_list=iter_list,tune_method="adapt_cov",par_type=par_type,par_setting={"dim":dim})

        obj.set_val(cov_tensor)
        params_obj_dict.update({"cov": obj})
    return(params_obj_dict)



