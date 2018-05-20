#from adapt_util.tune_param_classes.tune_param_class import tune_params_obj_creator
import torch,numpy,GPyOpt,math
from adapt_util.adapt_util import welford_tensor
from adapt_util.objective_funs_util import get_objective_fun
# can be called in fast,medium,slow state
class dual_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,update_iter_list,dual_param_objs_dict):
        #if len(dual_dict)>0:
        #    self.param_dict = dual_dict
        #else:
        #    raise ValueError("dual dict empty")
        self.update_iter_list = update_iter_list
        self.dual_param_objs_dict = dual_param_objs_dict
        #self.start_iter = self.update_iter_list[0]
        #self.end_iter = self.update_iter_list[-1]
        # dual_dict should be sorted by update priorities
        #self.dual_dict = dual_dict
        #self.dual_obj_dict = {}
        #for param,par_type in dual_dict.items():
        #    this_iter_list = self.update_iter_list
        #    if param in dual_param_obj_dict:
        #        self.dual_obj_dict.update({param:dual_param_obj_dict[param]})
            #else:
                #self.dual_obj_dict.update({param:tune_param_creator(param,this_iter_list,"dual",par_type)})
        #self.settings_dict = dual_settings()
        #print(self.update_iter_list)
        #exit()
        self.start_iter = self.update_iter_list[0]
        self.end_iter = self.update_iter_list[-1]
        # shared by different param_objs
        self.store_samples = []



    def initialize(self):
        self.next_refresh_iter = self.update_iter_list[1]
        self.cur_in_iter_list = 0
        for param,obj in self.dual_param_objs_dict.items():
            obj.initialize_tuning_param()
            obj.dual_metadata.initialize(init=obj.get_val())
        return()



    def update(self,sample_dict):
        iter = sample_dict["iter"]
        #print("iter is {}".format(iter))
        if iter < self.start_iter:
            pass
        elif iter >= self.end_iter:
            pass
        elif iter == self.start_iter:
            self.initialize()
            self.cur_in_iter_list += 1
        elif iter < self.next_refresh_iter:
            self.store_samples.append(sample_dict)

        elif iter == self.next_refresh_iter:
            print("reach here ")
            self.store_samples.append(sample_dict)
            for param, obj in self.dual_param_objs_dict.items():
                #print(self.store_samples[0])
                #print(sample_obj["log"].keys())
                #print(sample_obj["log"].store["accept_rate"])
                #print()
                #exit()
                objective = obj.dual_metadata.objective_fun(self.store_samples)
                #obj.dual_metadata.store_objective.append(objective)
                #print(obj.dual_metadata.__dict__)
                #print(dir(obj.dual_metadata))
                #exit()
                next_tune_par_val = obj.dual_metadata.update(objective)
                #print(next_tune_par_vals_dict)
                #exit()
                self.dual_param_objs_dict[param].set_val(next_tune_par_val)
            if self.update_iter_list[self.cur_in_iter_list]==self.end_iter:
                # end of update period.
                # do not need to update nex_iter if we are on our last point
                pass
            else:
                print("here too")
                obj.dual_metadata.cur_in_iter_list +=1
                self.cur_in_iter_list += 1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]
                # renew store samples
            self.store_samples = []
        else:
            print("iter is {}".format(iter))
            print("nex_refresh {}".format(self.next_refresh_iter))
            raise ValueError("shouldn't reach here")


# dual_metadata_argument(name,target,gamma,t_0,kappa,max_second_per_sample,obj_fun)
# opt_metadata_argument(obj_fun)
# each param_obj of tune method dual has an independent copy
class dual_param_metadata(object):
    def __init__(self,setting):
        self.target = setting["target"]
        self.par_type = setting["par_type"]
        self.gamma = setting["gamma"]
        self.t_0 = setting["t_0"]
        self.kappa = setting["kappa"]
        self.objective_fun = get_objective_fun(setting["obj_fun"])
        self.name = setting["name"]
        #self.obj = obj
        #print(self.objective_fun)
        #exit()
        # bar_ep_i and bar_H_i initialization does not matter as they are updated first
        self.bar_ep_i = 1
        self.bar_H_i = 0

        self.store_objective = []
        self.cur_in_iter_list = None
        self.store = []
    def initialize(self,init):
        self.cur_in_iter_list = 0
        self.mu = math.log(10 * init)
        return()
    def update(self, objective):
        # objective = objective fun evaluated for this samples in this window
        #print("reach here")
        print("objective fun value is {}".format(objective))
        print("target {}".format(self.target))
        self.bar_H_i = (1 - 1 / (self.cur_in_iter_list + 1 + self.t_0)) * self.bar_H_i + (
                1 / (self.cur_in_iter_list + 1 + self.t_0)) * (self.target-objective)
        logep = self.mu - math.sqrt(self.cur_in_iter_list + 1) / self.gamma * self.bar_H_i
        logbarep = math.pow(self.cur_in_iter_list + 1, -self.kappa) * logep + (
                1 - math.pow(self.cur_in_iter_list + 1, -self.kappa)) * math.log(
            self.bar_ep_i)
        self.bar_ep_i = math.exp(logbarep)
        self.store.append(self.bar_ep_i)
        #self.obj.set_val(self.bar_ep_i)
        self.cur_in_iter_list += 1
        return(self.bar_ep_i)
        # each param in dual_dict might have different objective functions

    def compute_objective(self,store_samples):
        objective = self.objective_fun(store_samples)
        self.store_objective.append(objective)
        return (objective)





# shared by all opt variables with the same par_type
class gpyopt_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,update_iter_list,opt_param_objs_dict=None,tune_setting=None):
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
        self.objective_fun = tune_setting.objective_fun
        self.opt_param_objs_dict = opt_param_objs_dict
        #self.store_object= []
        self.store_objective = []
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
        self.name_list = []
        for param_name,param_obj in self.opt_param_objs_dict:
            #start_temp.append(param.find_reasonable_start())
            start_temp.append(param_obj.get_val())
            self.name_list.append(param_name)
            bounds_temp.append({'name':param_name,'type':"continuous",'domain':param_obj.find_bounds()})
        self.X_step = numpy.array([start_temp])
        self.Y_step = None
        self.bounds = bounds_temp


    def compute_objective(self):
        out = self.objective_fun(self.store_samples)
        return(out)

    def update_gp(self,objective):
        if self.Y_step is None:
            self.Y_step = numpy.array([[objective]])
        else:
            self.Y_step = numpy.vstack((self.Y_step, numpy.array([[objective]])))
        bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, X=self.X_step, Y=self.Y_step)
        x_next = bo_step.suggest_next_locations()
        self.X_step = numpy.vstack((self.X_step, x_next))
        out_dict = {}
        for i in range(self.name_list):
            out_dict.update({self.name_list[i]:bo_step[i]})
        return(out_dict)
    def update(self,sample_dict):
        iter = sample_dict["iter"]
        if iter < self.start_iter:
            pass
        if iter >= self.end_iter:
            pass
        if iter == self.start_iter:
            self.initialize()
            self.cur_in_iter_list += 1
        if iter < self.next_refresh_iter:
            self.store_samples.append(sample_dict)
        if iter == self.next_refresh_iter:
            self.store_samples.append(sample_dict)
            objective = self.compute_objective()
            self.store_objective.append(objective)
            # do not need to explore next point if we are on our last point
            if self.update_iter_list[self.cur_in_iter_list]==self.end_iter:
                # when it is end of iteration set params to best value so far
                index = numpy.argmin(self.store_objective)
                best_X = self.X_step[index]
                for i in range(self.name_list):
                    self.opt_param_objs_dict[self.name_list[i]].set_val(best_X[i])
                pass
            else:
                next_tune_par_vals_dict = self.update_gp(objective)
                for param, obj in self.opt_param_objs_dict:
                    #obj.store = next_tune_par_vals_dict[param]
                    obj.set_val(next_tune_par_vals_dict[param])
                self.cur_in_iter_list +=1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]
            self.store = []


# used by the cov variable only, can be an par_type
class adapt_cov_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,update_iter_list,adapt_cov_param_objs_dict):
        #print(update_iter_list)
        #print(adapt_cov_param_objs_dict)
        #print(list(adapt_cov_param_objs_dict.items())[0])
        #exit()
        self.update_iter_list = update_iter_list
        self.cur_in_iter_list = None
        # opt_dict should be sorted by update priorities
        self.param_name,self.param_obj = list(adapt_cov_param_objs_dict.items())[0]
        self.start_iter = self.update_iter_list[0]
        self.end_iter = self.update_iter_list[-1]

        self.m_ = torch.zeros(self.param_obj.dim)
        if (self.param_name=="diag_cov"):
            self.m_2 = torch.zeros(self.param_obj.dim)
            self.diag = True
        elif(self.param_name=="dense_cov"):
            self.m_2 = torch.zeros(self.param_obj.dim,self.param_obj.dim)
            self.diag = False

    def initialize(self):
        self.next_refresh_iter = self.update_iter_list[1]
        self.cur_in_iter_list = 0
        self.iter = 0
    def update_cov(self,sample_dict):
        self.iter +=1
        self.m_, self.m_2 = welford_tensor(sample_dict["q"].flattened_tensor,self.iter,self.m_, self.m_2, self.diag)
        return()
    def refresh_cov(self):
        self.m_.zero_()
        self.m_2.zero_()
        self.iter = 0
        return()
    def update(self,sample_dict):

        iter = sample_dict["iter"]
        if iter < self.start_iter:
            pass
        if iter >= self.end_iter:
            pass
        if iter == self.start_iter:
            self.initialize()
            self.cur_in_iter_list += 1
        if iter < self.next_refresh_iter:
            pass
        if iter == self.next_refresh_iter:
            self.update_cov(sample_dict)
            #self.tuning_obj.integrator.set_metric(self.m_2)
            self.param_obj.set_val(self.m_2)
            # do not need to explore next point if we are on our last point
            if self.cur_in_iter_list==self.end_iter:
                pass
            else:
                self.refresh_cov()
                self.cur_in_iter_list +=1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]



class par_type_state(object):
    def __init__(self,par_type,update_iter_list,dual_objs_dict=None,opt_objs_dict=None,adapt_cov_objs_dict=None):
        self.par_type = par_type
        self.iter_list = update_iter_list

        if self.par_type=="fast":
            #print("yes")
            #print(dual_obj_dict)
            #exit()
            if len(dual_objs_dict)>0:
                #self.dual_state = dual_state(update_iter_list,dual_obj_dict)
                self.dual_state = dual_state(update_iter_list=update_iter_list,dual_param_objs_dict=dual_objs_dict)
                #print("yes")
                #exit()
        else:
            if len(dual_objs_dict)>0:
                self.dual_state = dual_state(update_iter_list=update_iter_list,dual_param_objs_dict=dual_objs_dict)
            if len(opt_objs_dict)>0:
                self.opt_state = gpyopt_state(update_iter_list=update_iter_list,opt_param_objs_dict=opt_objs_dict)
            if len(adapt_cov_objs_dict)>0:
                self.adapt_cov_state = adapt_cov_state(update_iter_list=update_iter_list,adapt_cov_param_objs_dict=adapt_cov_objs_dict)



# class diagnostics(object):
#     def __init__(self,dual_state_obj=None,opt_state_obj=None,adapt_state_obj=None):
#         self.dual_state = dual_state_obj
#         self.opt_state = opt_state_obj
#         self.adapt_state = adapt_state_obj
class tuning_param_states(object):
    def __init__(self,adapter,param_objs_dict):
        #dual_state_obj = None
        #opt_state_obj = None
        #adapt_state_obj = None
        self.slow_state = None
        self.medium_state = None
        self.fast_state = None

        self.params_obj_dict = None
        #self.params_obj_dict = tune_params_obj_creator(adapter.tune_dict, adapter)
        fast_dict,medium_dict,slow_dict = sort_objs_by_par_type_method(param_objs_dict)
        if len(slow_dict)>0:
            slow_dual_objs_dict,slow_opt_objs_dict,slow_adapt_cov_objs_dict = sort_objs_by_tune_method(slow_dict)
            self.slow_state = par_type_state(par_type="slow",update_iter_list=adapter.update_slow_list,
                                             dual_objs_dict=slow_dual_objs_dict,opt_objs_dict=slow_opt_objs_dict,
                                             adapt_cov_objs_dict=slow_adapt_cov_objs_dict)

        if len(medium_dict)>0:
            medium_dual_objs_dict, medium_opt_objs_dict, medium_adapt_cov_objs_dict =sort_objs_by_tune_method(medium_dict)
            self.medium_state = par_type_state(par_type="medium", update_iter_list=adapter.update_medium_list,
                                               dual_objs_dict=medium_dual_objs_dict, opt_objs_dict=medium_opt_objs_dict,
                                               adapt_cov_objs_dict=medium_adapt_cov_objs_dict)
        if len(fast_dict)>0:
            #print("yes")
            #print(adapter.update_fast_list)
            #exit()
            fast_dual_objs_dict, fast_opt_objs_dict, fast_adapt_cov_objs_dict=sort_objs_by_tune_method(fast_dict)
            #self.fast_state = par_type_state("fast",adapter.update_fast_list,fast_dual_objs_dict,
            #                                 fast_opt_objs_dict,fast_adapt_objs_dict)
            self.fast_state = par_type_state(par_type="fast", update_iter_list=adapter.update_fast_list,
                                             dual_objs_dict=fast_dual_objs_dict, opt_objs_dict=fast_opt_objs_dict,
                                             adapt_cov_objs_dict=fast_adapt_cov_objs_dict)
        #adapter.tuning_param_states = self





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
    adapt_cov_dict = {}
    opt_dict = {}
    dual_dict = {}
    for param, obj in param_obj_dict.items():
        if obj.tune_method=="adapt_cov":
            adapt_cov_dict.update({param:obj})
        if obj.tune_method=="opt":
            opt_dict.update({param:obj})
        if obj.tune_method=="dual":
            dual_dict.update({param:obj})
    return(dual_dict,opt_dict,adapt_cov_dict)
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
