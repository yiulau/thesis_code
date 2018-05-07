from adapt_util.adapt_util import tuneable_param,welford_tensor
from adapt_util.return_update_list import return_update_lists
import GPyOpt,numpy,torch
class adapter(object):
    #
    def __init__(self,one_chain_obj):

        self.one_chain_experiment = one_chain_obj
        # fast = tuned by dual. update every iter in ini_buffer and end_buffer
        # medium = tuned by dual or opt. update every window_size
        # slow = tuned by dual,adapt or opt, update every cur_window_size, which doubles after each update
        # default definitions
        self.par_type_dict = {"ep":"fast","L": "medium", "t": "medium", "alpha": "medium", "xhmc_delta": "medium",
                         "diag_cov": "slow", "cov": "slow"}
        self.tuneable_param = self.one_chain_experiment.tuneable_param
        self.dynamic = self.one_chain_experiment.dynamic
        self.second_order = self.one_chain_experiment.second_order
        self.metric = self.one_chain_experiment.Ham.T.metric.name
        self.criterion = self.one_chain_experiment.integrator.criterion
        # unique for individual sampler
        if self.dynamic==False:
            if not self.criterion is None:
                raise ValueError("static integrator should not have termination criterion")
        self.tuning_obj = None
        self.param_obj_dict = {}
        ep_obj = self.param_obj_dict["ep"]
        if self.ep_obj.tune_method=="opt":
            self.par_type_dict.update({"ep":"medium"})


    # what is status
    # status one of {fast,medium,slow}
    def add_param_dual(self,param,status):
        self.dual_dict.update({param:status})

    def add_param_opt(self,param,status):
        self.opt_dict.update({param:status})

    def add_param_adapt(self,param,status):
        self.adapt_dict.update({param:status})

    def add_param_fast(self,param,tune_method):
        if not tune_method=="fixed":
            self.fast_dict.update({param:tune_method})

    def add_param_medium(self,param,tune_method):
        if not tune_method=="fixed":
            self.medium_dict.update({param:tune_method})

    def add_param_slow(self,param,tune_method):
        if not tune_method=="fixed":
            self.slow_dict.update({param:tune_method})

    # for each
    # first determine if there are fast parameters
    def update_list(self):
        if len(self.fast_dict)>0:
            self.tune_fast=True
        if len(self.medium_dict)>0:
            self.tune_medium=True
        if len(self.slow_dict)>0:
            self.tune_slow=True

        out = return_update_lists(self.tune_l, self.tune_fast, self.tune_medium, self.tune_slow,
                                  self.ini_buffer,self.end_buffer,self.window_size,self.min_medium_updates,
                                  self.min_slow_updates)

        self.update_fast_list = out[0]
        self.update_medium_list = out[1]
        self.update_fast_list = out[2]


    def update(self,sample_obj):
        # by definition
        if self.tune_fast:
            for param,status in self.dual_dict:
                self.param_obj_dict[param].update(sample_obj,self.update_fast_list)

        if self.tune_medium:
            # some could be tuned by opt, other by dual
            for param,status in self.dual_dict:
                self.param_obj_dict[param].update(sample_obj,self.update_medium_list)
            if len(self.opt_dict)>0:
                self.gpyopt_obj.update(sample_obj,self.update_medium_list)
        if self.tune_slow:
            for param,status in self.slow_dict:
                self.param_obj_dict[param].update(sample_obj,self.update_slow_list)



class gpyopt_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,opt_dict,update_iter_list):
        if len(opt_dict)>0:
            self.param_dict = opt_dict
        else:
            raise ValueError("opt dict empty")
        self.update_iter_list = update_iter_list
        self.cur_in_iter_list = None
        # opt_dict should be sorted by update priorities
        self.opt_dict = opt_dict
        self.start_iter = self.update_iter_list[0]
        self.end_iter = self.update_iter_list[-1]
        self.store = []
        self.objective_fun


    def initialize(self):
        self.next_refresh_iter = self.update_iter_list[1]
        self.cur_in_iter_list = 0
        start_temp = []
        bounds_temp = []
        for param in self.opt_dict:
            start_temp.append(param.find_reasonable_start())
            bounds_temp.append({'name':param.name,'type':"continuous",'domain':param.find_bounds()})
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
            self.store.append(sample_obj)
        if iter == self.next_refresh_iter:
            self.store.append(sample_obj)
            objective = self.compute_objective()
            self.store_obj.append(objective)
            # do not need to explore next point if we are on our last point
            if self.cur_in_iter_list==self.end_iter:
                pass
            else:
                next_tune_par_vals_dict = self.update_gp(objective)
                for param, status in self.opt_dict:
                    self.param_dict[param] = next_tune_par_vals_dict[param]
                self.store = []
                self.cur_in_iter_list +=1
                self.next_refresh_iter = self.update_iter_list[self.cur_in_iter_list]


class adapt_cov_state(object):
    # only initialize when opt_dict not empty . upstream decision
    def __init__(self,param,update_iter_list):

        self.update_iter_list = update_iter_list
        self.cur_in_iter_list = None
        # opt_dict should be sorted by update priorities
        self.param = param
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


 # adapt_option one of {1,2,3,4,5,6}
        # adapt_cov: true or false
        # option1: adapt epsilon only (NUTS-unit_e) by dual averaging
        # straightforward - already implemented
        # option2 : adapt both epsilon (dual averaging) and Cov or cov_diag (NUTS- dense_e,diag_e)
        # also already implemented
        # option 3: gpyopt choose (epsilon,T) option adapt cov or cov_diag(HMC-unit_e,dense_e,diag_e)
        # equal-length windows collecting esjd points for (ep,t), update metric at the end of each window
        # first window is unit_e by default, unless there is something to initialize it
        # (epsilon,T) initialized by find_reasonable_ep and find_reasonable_t (find_reasonable_ep * 3.1 e.g. ,
        # something that makes a short trajectory
        # option 4: gpyopt choose epsilon option to adapt cov or cov_diag (NUTS - dense_e,diag_e,unit_e)
        # start with initialization chose by find_reasonable_ep
        # equally divided windows . update cov at end of window
        # option 5: adapt epsilon (dual_averaging) then gpyopt choose delta option adapt cov or cov_diag(xhmc- unit_e, dense_e,diag_e)
        # option 6: gpyopt choose (epsilon,delta) option adapt cov or cov_diag (xhmc - unit_e, diag_e,dense_e)
        # option 7: adapt epsilon by dual averaging, adapt_t by GPyOpt

