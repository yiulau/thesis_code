from adapt_util.tune_param_classes.tune_param_class import tune_params_obj_creator
import torch,numpy,GPyOpt,math

class dual_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,dual_dict,update_iter_list,dual_param_obj_dict=None):
        if len(dual_dict)>0:
            self.param_dict = dual_dict
        else:
            raise ValueError("dual dict empty")
        self.update_iter_list = update_iter_list

        # dual_dict should be sorted by update priorities
        self.dual_dict = dual_dict
        self.dual_obj_dict = {}
        for param,par_type in dual_dict.items():
            this_iter_list = self.fast_list
            if param in dual_param_obj_dict:
                self.dual_obj_dict.update({param:dual_param_obj_dict[param]})
            #else:
                #self.dual_obj_dict.update({param:tune_param_creator(param,this_iter_list,"dual",par_type)})
        #self.settings_dict = dual_settings()
        #self.start_iter = self.update_iter_list[0]
        #self.end_iter = self.update_iter_list[-1]
        self.store_samples = []



    def initialize(self):
        for param,obj in self.dual_obj_dict.items():
            obj.initialize()
        return()

    #each param in dual_dict might have different objective functions
    def compute_objective(self):
        out = self.objective_fun(self.store)
        return(out)

    def update(self,sample_obj):
        iter = sample_obj.iter
        if iter < self.start_iter:
            pass
        if iter >= self.end_iter:
            pass
        if iter == self.start_iter:
            self.initialize()

        if iter < self.next_refresh_iter:
            self.store_samples.append(sample_obj)
        if iter == self.next_refresh_iter:
            self.store_samples.append(sample_obj)

            if self.cur_in_iter_list==self.end_iter:
                pass
            else:
                for param, obj in self.dual_obj_dict:
                    objective = obj.compute_objective(sample_obj)
                    obj.store_objective.append(objective)
                    # do not need to explore next point if we are on our last point
                    next_tune_par_vals_dict = obj.update_dual(objective)
                    self.param_objs_dict[param].set_val(next_tune_par_vals_dict[param])
                self.store_samples = []
                self.cur_in_iter_list +=1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]


# dual_metadata_argument(name,target,gamma,t_0,kappa,max_second_per_sample,obj_fun)
# opt_metadata_argument(obj_fun)
# each param_obj of tune method dual has an independent copy
class dual_param_metadata(object):
    def __init__(self,name,target=None,gamma=0.05,t_0=10,kappa=0.75,max_second_per_sample=0.1,obj_fun="accept_rate"):
        self.target = target
        self.gamma = gamma
        self.t_0 = t_0
        self.kappa = kappa
        self.max_second_per_sample = max_second_per_sample
        self.objective_fun = obj_fun
        # bar_ep_i and bar_H_i initialization does not matter as they are updated first
        self.bar_ep_i = 1
        self.bar_H_i = 0
        self.name = name
        self.cur_in_iter_list = 0
        if name=="epsilon" and self.target is None:
            self.target = 0.65


        elif self.target is None:
            # assuming objective function is normalized esjd
            self.target = 0.99

        def initialize(self,init):
            self.mu = math.log(10 * init)
            return()
        def update(self, input):
            self.bar_H_i = (1 - 1 / (self.cur_in_iter_list + 1 + self.t_0)) * self.bar_H_i + (
                    1 / (self.cur_in_iter_list + 1 + self.t_0)) * (
                               input)
            logep = self.mu - math.sqrt(self.cur_in_iter_list + 1) / self.gamma * self.bar_H_i
            logbarep = math.pow(self.cur_in_iter_list + 1, -self.kappa) * logep + (
                    1 - math.pow(self.cur_in_iter_list + 1, -self.kappa)) * math.log(
                self.bar_ep_i)
            self.bar_ep_i = math.exp(logbarep)
            self.store.append(self.bar_ep_i)
            self.set_val(self.bar_ep_i)
            self.cur_in_iter_list += 1





# shared by all opt variables with the same par_type
class gpyopt_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,update_iter_list,param_obj_dict=None,metadata=None):
        #if len(opt_dict)>0:
        #    self.param_dict = opt_dict
        #else:
        #    raise ValueError("opt dict empty")
        self.update_iter_list = update_iter_list
        self.cur_in_iter_list = None
        # opt_dict should be sorted by update priorities
        #self.opt_dict = opt_dict
        self.start_iter = self.update_iter_list[0]
        self.end_iter = self.update_iter_list[-1]
        self.store_samples = []
        self.objective_fun = metadata.objective_fun
        self.opt_param_obj_dict = param_obj_dict
        self.store_object= []
        #self.opt_obj_dict = {}
        #for param, par_type in opt_dict:
        #    #this_iter_list =
        #    if param in opt_param_obj_dict:
        #        self.opt_obj_dict.update({param: opt_param_obj_dict[param]})
        #    else:
                #self.dual_obj_dict.update({param: tune_param_creator(param, this_iter_list, "opt", par_type)})

    def initialize(self):
        self.next_refresh_iter = self.update_iter_list[1]
        self.cur_in_iter_list = 0
        start_temp = []
        bounds_temp = []
        for param_obj in self.param_obj_dict:
            #start_temp.append(param.find_reasonable_start())
            start_temp.append(param_obj.get_val())
            bounds_temp.append({'name':param_obj.name,'type':"continuous",'domain':param_obj.find_bounds()})
        self.X_step = numpy.array([start_temp])
        self.Y_step = None
        self.bounds = bounds_temp


    def compute_objective(self):
        out = self.objective_fun(self.store)
        return(out)

    def update_gp(self,objective):
        if self.Y_step is None:
            self.Y_step = numpy.array([[objective]])
        else:
            self.Y_step = numpy.vstack((self.Y_step, numpy.array([[objective]])))
        bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, X=self.X_step, Y=self.Y_step)
        x_next = bo_step.suggest_next_locations()
        return(bo_step)
    def update(self,sample_obj):
        iter = sample_obj.iter
        if iter < self.start_iter:
            pass
        if iter >= self.end_iter:
            pass
        if iter == self.start_iter:
            self.initialize()

        if iter < self.next_refresh_iter:
            self.store_samples.append(sample_obj)
        if iter == self.next_refresh_iter:
            self.store_samples.append(sample_obj)
            objective = self.compute_objective()
            self.store_objective.append(objective)
            # do not need to explore next point if we are on our last point
            if self.cur_in_iter_list==self.end_iter:
                pass
            else:
                next_tune_par_vals_dict = self.update_gp(objective)
                for param, obj in self.opt_param_obj_dict:
                    #obj.store = next_tune_par_vals_dict[param]
                    obj.set_val(next_tune_par_vals_dict[param])
                self.store = []
                self.cur_in_iter_list +=1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]


# used by the cov variable only, can be an par_type
class adapt_cov_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,param_obj,update_iter_list):

        self.update_iter_list = update_iter_list
        self.cur_in_iter_list = None
        # opt_dict should be sorted by update priorities
        self.param_obj = param_obj
        self.start_iter = self.update_iter_list[0]
        self.end_iter = self.update_iter_list[-1]

        self.m_ = torch.zeros(self.param.dim)
        if (self.param.name=="diag_cov"):
            self.m_2 = torch.zeros(self.param.dim)
            self.diag = True
        else:
            self.m_2 = torch.zeros(self.param.dim,self.param_dim)
            self.diag = False

    def initialize(self):
        self.next_refresh_iter = self.update_iter_list[1]
        self.cur_in_iter_list = 0
        self.iter = 0
    def update_cov(self,sample_obj):
        self.iter +=1
        self.m_, self.m_2 = welford_tensor(sample_obj.q.flattened_tensor,self.iter,self.m_, self.m_2, self.diag)
        return()
    def refresh_cov(self):
        self.m_.zero_()
        self.m_2.zero_()
        self.iter = 0
        return()
    def update(self,sample_obj):
        iter = sample_obj.iter
        if iter < self.start_iter:
            pass
        if iter >= self.end_iter:
            pass
        if iter == self.start_iter:
            self.initialize()

        if iter < self.next_refresh_iter:
            pass
        if iter == self.next_refresh_iter:
            self.update_cov(sample_obj)
            self.tuning_obj.integrator.set_metric(self.m_2)
            # do not need to explore next point if we are on our last point
            if self.cur_in_iter_list==self.end_iter:
                pass
            else:
                self.refresh_cov()
                self.cur_in_iter_list +=1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]



class par_type_state(object):
    def __init__(self,name,update_iter_list,dual_obj_dict=None,opt_obj_dict=None,adapt_obj_dict=None):
        self.par_type = name
        self.iter_list = update_iter_list
        if self.par_type=="fast":
            if len(dual_obj_dict)>0:
                self.dual_state = dual_state(dual_obj_dict,update_iter_list)
        else:
            if len(dual_obj_dict)>0:
                self.dual_state = dual_state(dual_obj_dict,update_iter_list)
            if len(opt_obj_dict)>0:
                self.opt_state = gpyopt_state(opt_obj_dict,update_iter_list)
            if len(adapt_obj_dict)>0:
                self.dual_state = dual_state(adapt_obj_dict,update_iter_list)



class diagnostics(object):
    def __init__(self,dual_state_obj=None,opt_state_obj=None,adapt_state_obj=None):
        self.dual_state = dual_state_obj
        self.opt_state = opt_state_obj
        self.adapt_state = adapt_state_obj
class tuning_param_states(object):
    def __init__(self,adapter):
        #dual_state_obj = None
        #opt_state_obj = None
        #adapt_state_obj = None
        self.slow_state = None
        self.medium_state = None
        self.fast_state = None

        self.params_obj_dict = tune_params_obj_creator(adapter.tune_dict, adapter)
        fast_dict,medium_dict,slow_dict = sort_objs_by_par_type_method(self.params_obj_dict)
        if len(slow_dict)>0:
            slow_dual_objs_dict,slow_opt_objs_dict,slow_adapt_objs_dict = sort_objs_by_tune_method(slow_dict)
            self.slow_state = par_type_state("slow",adapter.slow_update_list,slow_dual_objs_dict,slow_opt_objs_dict,
                                             slow_adapt_objs_dict)
        if len(medium_dict)>0:
            medium_dual_objs_dict, medium_opt_objs_dict, medium_adapt_objs_dict =sort_objs_by_tune_method(medium_dict)
            self.medium_state = par_type_state("medium",adapter.medium_update_list,medium_dual_objs_dict,
                                               medium_opt_objs_dict,medium_adapt_objs_dict)
        if len(fast_dict)>0:
            fast_dual_objs_dict, fast_opt_objs_dict, fast_adapt_objs_dict=sort_objs_by_tune_method(fast_dict)
            self.fast_state = par_type_state("fast",adapter.update_fast_list,fast_dual_objs_dict,
                                             fast_opt_objs_dict,fast_adapt_objs_dict)

        adapter.tuning_param_states = self





       #  if len(adapter.dual_dict)>0:
       #     dual_state_obj = dual_state(adapter.dual_dict,adapter.fast_update_list,
       #                                 adapter.medium_update_list,adapter.slow_update_list,
       #                                 adapter.dual_param_obj_dict)
       #  if len(adapter.opt_dict)>0:
       #     opt_state_obj = gpyopt_state(adapter.opt_dict,adapter.fast_update_list,
       #                                 adapter.medium_update_list,adapter.slow_update_list,
       #                                 adapter.opt_param_obj_dict)
       #  if len(adapter.adapt_dict)>0:
       #     adapt_cov_state_obj = adapt_cov_state(self.params_obj_dict["cov"],
       #                                           adapter.slow_update_list)
       #
       # self.diagnostics = diagnostics(dual_state_obj,opt_state_obj,adapt_cov_state_obj)






def sort_objs_by_tune_method(param_obj_dict):
    adapt_dict = {}
    opt_dict = {}
    dual_dict = {}
    for param, obj in param_obj_dict.items():
        if obj.tune_method=="adapt":
            adapt_dict.update({param:obj})
        if obj.tune_method=="opt":
            opt_dict.update({param:obj})
        if obj.tune_method=="dual":
            dual_dict.update({param:obj})
    return(dual_dict,opt_dict,adapt_dict)
def sort_objs_by_par_type_method(param_obj_dict):
    fast_dict = {}
    medium_dict = {}
    slow_dict = {}
    for param, obj in param_obj_dict.items():
        if obj.par_type=="fast":
            fast_dict.update({param:obj})
        if obj.par_type=="medium":
            medium_dict.update({param:obj})
        if obj.par_type=="slow":
            slow_dict.update({param:obj})
    return(fast_dict,medium_dict,slow_dict)
