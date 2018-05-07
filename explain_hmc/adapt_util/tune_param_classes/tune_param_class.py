import abc,math, torch
from adapt_util.find_reasonable_start import find_reasonable_ep
class tune_param(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,update_iter_list,tune_method,param_type,tuning_obj):
        self.update_iter_list = update_iter_list
        self.tune_method = tune_method
        self.parma_type = param_type
        self.tuning_obj = tuning_obj
        self.iter = 0
        # tuning_obj should have diagnostics_obj which contains gpyopt obj, which is the shared gpyopt state for
        # all params tuned by opt
        # tuning obj should also contain param_obj_dict which contains all the param objects in a dictionary
        # with its name as key

    def update_dual(self,input):
        bar_H_i = (1 - 1 / (self.iter + 1 + self.t_0)) * self.bar_H_i + (1 / (self.iter + 1 + self.t_0)) * (
                input)
        logep = self.mu - math.sqrt(self.iter + 1) / self.gamma * self.bar_H_i
        logbarep = math.pow(self.iter + 1, -self.kappa) * logep + (1 - math.pow(self.iter + 1, -self.kappa)) * math.log(
            self.bar_ep_i)
        self.bar_ep_i = math.exp(logbarep)
        self.store.append(self.bar_ep_i)
        self.cur_val = self.bar_ep_i
        self.iter += 1



    @abc.abstractmethod
    def find_reasonable_start(self):
        # returns reasonable initialization value
        return()

    @abc.abstractmethod
    def update_internal_state(self,sample_obj):
        # takes one sample object - storing relevant information about the current transition
        # and update the internal object
        return()




class epsilon(tune_param):
    def __init__(self,update_iter_list,tune_method,param_type):
        super(epsilon, self).__init__(update_iter_list,tune_method,param_type)
        self.update_priority = 1
        if self.tune_method=="dual":
            # initialize relevant statistics for dual averaging
            self.settings = self.dual_settings["ep"]
            ep = find_reasonable_ep()
            self.target_delta = self.settings.target_delta
            self.gamma = self.settings.gamma
            self.t_0 = self.settings.t_0
            self.mu = math.log(10 * ep)
            self.bar_ep_i = 1
            self.bar_H_i = 0
            self.cur_val = None
            self.bounds = (1e-6,0.5)
            pass
        elif self.tune_method=="opt":
            self.cur_val =
            self.gpyopt_obj = self.tuning_obj.diagnostics_obj.gpyopt_obj
            self.bounds = (1e-6,0.5)
            pass
            # initialize ep for bayesian optimization
        elif self.tune_method=="fixed":
            self.cur_val =

            pass
        else:
            raise ValueError("unknown tune method")

    def find_reasonable_start(self):
        # unit_e unit trajectory length
        epsilon = find_reasonable_ep(self.tuning_obj)
        return(epsilon)
    def update_internal_state(self,sample_obj):
        if self.tune_method=="dual":
            alpha = sample_obj.accept_rate
            input = self.target_delta - alpha
            self.update_dual(input)
        elif self.tune_method=="opt":
            self.gpyopt_obj.update(sample_obj)
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method ")
    def find_bounds(self):
        if self.tune_method=="fixed":
            self.bounds = (self.cur_val*0.9,self.cur_val*1.1)
        return(self.bounds)

class evolve_t(tune_param):
    def __init__(self,update_iter_list,tune_method,param_type):
        super(evolve_t, self).__init__(update_iter_list,tune_method,param_type)
        self.update_priority = 2
        self.param_obj_dict = self.tuning_obj.param_obj_dict
        if self.tune_method=="dual":
            # initialize relevant statistics for dual averaging
            input =
            self.update_dual(input)
            pass
        elif self.tune_method=="opt":
            self.gpyopt_obj = self.diagnostics_obj.gpyopt_obj

            pass
            # initialize ep for bayesian optimization
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method")
        self.ave_second_per_leapfrog = self.integrator.find_ave_second_per_leapfrog()
        self.max_second_per_sample = self.settings.max_second_per_sample
        self.bounds = self.find_bounds()
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        out = (self.bounds[0] + self.bounds[1])*0.5
        return(epsilon)
    def update_internal_state(self,sample_obj):
        if self.tune_method=="dual":
            pass
        elif self.tune_method=="opt":
            self.gpyopt_obj.update(sample_obj)
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method ")
    def find_bounds(self):
        ep_bounds = self.tuning_object.param_dict["ep"].find_bounds()
        max_t = (self.maximum_second_per_sample / self.ave_second_per_leapfrog) * ep_bounds[1]
        min_t = ep_bounds[1] * 3.1
        self.bounds = (min_t,max_t)
        return(ep_bounds)

class evolve_L(tune_param):
    def __init__(self,update_iter_list,tune_method,param_type,param_obj_dict):
        super(evolve_L, self).__init__(update_iter_list,tune_method,param_type)
        self.update_priority = 2
        self.param_obj_dict = param_obj_dict
        if self.tune_method=="dual":
            # initialize relevant statistics for dual averaging
            pass
        elif self.tune_method=="opt":
            self.gpyopt_obj = self.diagnostics_obj.gpyopt_obj
            self.bounds = self.find_bounds()
            pass
            # initialize ep for bayesian optimization
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method")
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        out = round((self.bounds[0]+self.bounds[1])*0.5)
        return(epsilon)
    def update_internal_state(self,sample_obj):
        if self.tune_method=="dual":
            pass
        elif self.tune_method=="opt":
            self.gpyopt_obj.update(sample_obj)
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method ")

    def find_bounds(self):
        ep_bounds = self.tuning_object.param_dict["ep"].find_bounds()
        max_L = round((self.maximum_second_per_sample / self.ave_second_per_leapfrog) + 1 )
        min_L = 1
        self.bounds = (min_L, max_L)
        return(self.bounds)


class alpha(tune_param):
    def __init__(self,update_iter_list,tune_method,param_type,tuning_obj):
        super(alpha, self).__init__(update_iter_list,tune_method,param_type)
        self.update_priority = 3
        self.param_obj_dict = self.tuning_obj.param_obj_dict
        if self.tune_method=="dual":
            # initialize relevant statistics for dual averaging
            pass
        elif self.tune_method=="opt":
            self.gpyopt_obj = self.diagnostics_obj.gpyopt_obj
            self.bounds
            pass
            # initialize ep for bayesian optimization
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method")
        self.bounds = (1e-6,1e6)
    def find_reasonable_start(self):
        # unit_e unit trajectory length
        reasonable_start = 1
        return(epsilon)
    def update_internal_state(self,sample_obj):
        if self.tune_method=="dual":
            pass
        elif self.tune_method=="opt":
            self.gpyopt_obj.update(sample_obj)
        elif self.tune_method=="fixed":
            pass
        else:
            raise ValueError("unknown tune method ")

    def find_bounds(self):
        return(self.bounds)



class xhmc_delta(tune_param):
    def __init__(self, update_iter_list, tune_method, param_type,tuning_obj):
        super(xhmc_delta, self).__init__(update_iter_list, tune_method,param_type, tuning_obj)
        self.update_priority = 4

        if self.tune_method == "dual":
            # initialize relevant statistics for dual averaging
            pass
        elif self.tune_method == "opt":
            self.gpyopt_obj = self.diagnostics_obj.gpyopt_obj

            pass
            # initialize ep for bayesian optimization
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method")
        self.bounds = (1e-6,0.2)

    def find_reasonable_start(self):
        # unit_e unit trajectory length
        reasonable_start = 0.1
        return (epsilon)

    def update_internal_state(self, sample_obj):
        if self.tune_method == "dual":
            pass
        elif self.tune_method == "opt":
            self.gpyopt_obj.update(sample_obj)
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method ")

    def find_bounds(self):
        out =
        self.bounds= out
        return(out)

class cov(tune_param):
    def __init__(self, update_iter_list, tune_method, param_type, tuning_obj):
        super(cov, self).__init__(update_iter_list, tune_method, param_type, tuning_obj)
        self.update_priority = 5
        self.store = torch.zeros(tuning_obj.dim,tuning_obj.dim)
        if self.tune_method == "adapt":
            # initialize relevant statistics for adapting
            self.adapt_obj = self.tuning_obj.diagnostics_obj.adapt_obj
            pass
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method")

    def find_reasonable_start(self):
        # unit matrix
        pass

    def update_internal_state(self, sample_obj):
        if self.tune_method == "adapt":
            self.adapt_obj.update(sample_obj)
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method ")

    def set_value(self,cov_tensor):
        self.store.copy_(cov_tensor)



class diag_cov(tune_param):
    def __init__(self, update_iter_list, tune_method, param_type, tuning_obj):
        super(diag_cov, self).__init__(update_iter_list, tune_method, param_type, tuning_obj)
        self.update_priority = 5
        self.store = torch.zeros(tuning_obj.dim)
        if self.tune_method == "adapt":
            self.adapt_obj = self.tuning_obj.diagnostics_obj.adapt_obj

        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method")



    def update_internal_state(self, sample_obj):
        if self.tune_method == "adapt":
            self.adapt_obj.update(sample_obj)
        elif self.tune_method == "fixed":
            pass
        else:
            raise ValueError("unknown tune method ")

    def set_value(self,diag_cov_tensor):
        self.store.copy_(diag_cov_tensor)


