from multiprocessing import Pool
import numpy,multiprocessing
from abstract.abstract_adapt_util import welford, adapt_ep, return_update_ep_list
import GPyOpt
from explicit.adapt_util import return_update_GPyOpt_list


# number of samples
# thinning
# warm up
# initialize at specific point
# number of chains
# parallel computing
# gpu



# delta is a slow parameter like cov and cov_diag, need to be upgraded slowly. consuming signifcant number of samples at
# each update
# integration time t is also a slow parameter, because diagnostics (ESS) for its performance can only be calculated by looking
# at a number of samples
class mcmc_sampler(object):

    def __init__(self,sampler_one_step,adapter,num_samples_per_chain,thinning,warm_up_per_chain,initialization,num_chains,parallel_computing,precision,isstore_to_disk):
        self.store_chains = numpy.empty(num_chains, object)
        self.parallel_computing = parallel_computing
        self.num_chains = num_chains

        return()

    def start_sampling(self):
        if self.parallel_computing:
            # do something parallel
            num_cpu =2
            with multiprocessing.Pool(processes=num_cpu) as pool:
                result_parallel = pool.map(one_chain_experiment.run, range(self.num_chains))

        else:
            result_sequential = [None]*self.num_chains
            for i in range(self.num_chains):
                result_sequential.append(one_chain_experiment.run())
                #experiment = one_chain_sampling(self.precision, self.initialization, self.sampler_one_step, self.adapter)

        return()

    def pre_experiment_diagnostics(self,test_run_chain_length=100):
        #

class sampling_metadata(object):
    def __init__(self,sampling_obj,id,num_samples,thin=1,experiment_obj=None):
        # parallel chains sampling from the same distribution shares sampling_obj
        # e.g. 4 chains sampling from iid normal
        # different chains sampling in the context of the same experiment shares experiment_obj
        # e.g. HMC(ep,t) sampling from a model with different values of (ep,t) on a grid
        # thin an integer >=0 skips
        # period for saving samples
        self.experiment_obj = experiment_obj
        self.sampling_obj = sampling_obj
        self.chain_id = id
        self.num_samples = num_samples
        self.thin_while_sampling = thin





class one_chain_obj(object):
    def __init__(self,Ham,windowed,precision,sampling_metadata=None,initialization=None,adapter=None):
        self.sampling_metadata = sampling_metadata
        self.precision = precision
        self.sampler_one_step = generate_sampler_one_step(Ham,windowed,precision)
        self.initialization = initialization
        self.adapter = adapter
        self.store_samples = numpy.empty()
        if self.sampling_metadata is None:
            self.sampling_metadata = sampling_metadata()
        if self.initialization is None:
            self.initialization = initialization()


    def adapt(self,out,counter):
        # if self.adapter is none, do nothing
        if self.adapter==None:
            pass
        else:
            out = self.adapter.update(out,counter)
            self.sampler_one_step = out.sampler_one_step

    def store_to_disk(self):
        pass

    def run(self):
        self.initialization()

        for counter in range(self.num_samples_per_chain):
            out = self.sampler_one_step()
            self.addsample(out, counter)
            if counter < self.warm_up_per_chain:
                self.adapt(out, counter)
            if self.isstore_to_disk(counter):
                self.store_to_disk()
        return()
    def addsample(self,out,counter):
class initialization(object):
    # should contain a point object
    def __init__(self,V,T,q=None,p=None):
        self.V = V
        self.T = T
        self.p = V.q_point
        self.q = T.p_point

    def load_point(self,q=None,p=None):
        if not q ==None:
            self.q.flattened_tensor = q.flattened_tensor
            self.q.load_flatten()
        if not p==None:
            self.p.flattened_tensor = p.flattened_tensor
            self.p.load_flatten()
        return()





class adapter(object):
    #
    def __init__(self,metric,adapt_option=None,adapt_epsilon=None,adapt_t=None,adapt_cov=None,adapt_delta=None,):
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
        self.metric = metric
        if adapt_option == 1:
            self.adapt_epsilon = "dual_averaging"
            self.adapt_time = "NUTS"
        elif adapt_option == 2:
            self.adapt_epsilon = "dual_averaging"
            self.adapt_cov = True
            self.adapt_time = "NUTS"
        elif adapt_option == 3:
            self.adapt_epsilon ="GPyOpt"
            self.adapt_time = "GPyOpt"
            self.adapt_cov = adapt_cov
        elif adapt_option == 4:
            self.adapt_epsilon ="GPyOpt"
            self.adapt_time = "NUTS"
        elif adapt_option == 5:
            self.adapt_epsilon = "dual_averaging"
            self.adapt_delta = "GPyOpt"
            self.adapt_cov =adapt_cov
            self.adapt_time = "NUTS"
        elif adapt_option == 6:
            self.adapt_epsilon = "GPyOpt"
            self.adapt_delta = "GPyOpt"
            self.adapt_cov = adapt_cov
            self.adapt_time = "NUTS"
        else:
            self.adapt_epsilon = adapt_option
            self.adapt_t = adapt_t
            self.adapt_cov = adapt_cov
            self.adapt_delta = adapt_delta

    def update(self,out,counter,metric):
        if counter in self.update_ep_list:
            if self.adapt_epsilon=="dual_averaging":
                alpha = out[3]
                bar_ep_i, self.bar_H_i,self.counter_ep = adapt_ep(alpha, self.bar_H_i, self.t_0, self.counter_ep, self.target_delta, self.gamma, self.bar_ep_i, self.kappa, self.mu)
                self.ep = bar_ep_i
            elif self.adapt_epsilon=="GPyOpt":
                self.counter_ep +=1
                feed_GPyOpt = True
                self.GPyOpt_machine.ep = self.ep


        if counter in self.update_cov_list:
            self.m_, self.m_2, self.counter_cov = welford(out.q.flattened_tensor, self.counter_cov, self.m_, self..m_2, metric.name)
            if counter in self.renew_cov_list:
                metric.set_metric(self.m_2)
        if counter in self.update_t_list:
            self.counter_t +=1
            feed_GPyOpt = True
            self.GPyOpt_machine.t = self.t
        if counter in self.update_delta_list:
            self.counter_delta +=1
            feed_GPyOpt = True
            self.GPyOpt_machine.delta = self.delta
        if feed_GPyOpt:

            self.feed_GPyOpt(out)

        else:
            pass

        self.GPyOpt_run()
        self.store_ep[counter] = self.ep

        return()

    def GPyOpt_initialize(self):

        if self.adapt_epsilon=="GPyOpt" and self.adapt_delta=="GPyOpt" :
            # can only happen in xhmc
            reasonable_start_ep = find_reasonable_start_ep()
            reasonable_start_delta = find_reasonable_start_delta(reasonable_start_ep)
            ep_bounds = find_epsilon_bounds(reasonable_start_ep,reasonable_start_delta) # use find_reasonable_start_ep
            delta_bounds = find_delta_bounds(reasonable_start_ep,reasonable_start_delta) # dependent on ep_bounds
            self.bounds = bounds =[{'name': 'ep', 'type': 'continuous', 'domain': (ep_bounds[0],ep_bounds[1])},
                                    {'name': 'delta', 'type': 'continuous', 'domain': (delta_bounds[0],delta_bounds[1])}]
            self.x.next = numpy.array([[reasonable_start_ep,reasonable_start_delta]])
        elif self.adapt_epsilon=="GPyOpt" and self.adapt_t=="GPyOpt:
            # happen in static hmc, static rhmc
            reasonable_start_ep = find_reasonable_start_ep()


            ep_bounds = find_epsilon_bounds(reasonable_start_ep)  # use find_reasonable_start_ep
            max_t = self.maximum_time_per_sample / ep_bounds[0]
            # at least integrates 3 steps
            min_t = ep_bounds[1] * 3.1
            reasonable_start_t = (max_t + min_t) * 0.5
            t_bounds = [min_t,max_t]  # dependent on ep_bounds
            self.bounds = bounds = [{'name': 'ep', 'type': 'continuous', 'domain': (ep_bounds[0], ep_bounds[1])},
                                    {'name': 'delta', 'type': 'continuous',
                                     'domain': (t_bounds[0], t_bounds[1])}]
            self.x.next = numpy.array([[reasonable_start_ep, reasonable_start_t]])
        elif self.adapt_t=="GPyOpt":
            # dual averaging adapt epsilon Gpyopt adapt t. slow variable. also ad or fixed ep
            max_t = self.maximum_time_per_sample / ep_bounds[0]
            # at least integrates 3 steps
            min_t = ep_bounds[1] * 3.1
            t_bounds = [min_t, max_t]  # dependent on ep_bounds
            reasonable_start_t = (max_t + min_t) * 0.5
            self.bounds = bounds = [{'name': 'delta', 'type': 'continuous','domain': (t_bounds[0], t_bounds[1])}]
            self.x.next = numpy.array([[ reasonable_start_t]])
        elif self.adapt_epsilon == "GPyOpt":
                # static case. fixed t . dynamic case .
                reasonable_start_ep = find_reasonable_start_ep()
                ep_bounds = find_epsilon_bounds(reasonable_start_ep)
                self.bounds = bounds = [{'name': 'ep', 'type': 'continuous', 'domain': (ep_bounds[0], ep_bounds[1])}]
                x.next =  numpy.array([[reasonable_start_ep]])
            elif self.adapt_delta == "GPyOpt:
                reasonable_start_delta = find_reasonable_start_delta()
                delta_bounds = find_delta_bounds(reasonable_start_delta)
                self.bounds = bounds = [{'name': 'delta', 'type': 'continuous', 'domain': (delta_bounds[0], delta_bounds[1])}]
                x.next = numpy.array([[reasonable_start_delta]])

    def GPyOpt_run(self):
        #order [ep,delta],[ep,t]
        # initialize if it has not happened
        if self.X_step == None and self.Y_step==None:
            self.GPyOpt_initialize()
            X_step = self.x.next
            esjd = ESJD(self.GPyOpt_counter)
            Y_step = numpy.array([[esjd]])
            bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, self.X_step, self.Y_step)
            x_next = bo_step.suggest_next_locations()
            self.x.next = x_next
            self.X_step = numpy.vstack((X_step, x_next))
        else:
            y_next = numpy.array([[self.ESJD(self.GPyOpt_counter)]])
            Y_step = numpy.vstack((self.Y_step, y_next))
            bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, self.X_step,self.Y_step)
            x_next = bo_step.suggest_next_locations()
            self.x.next = x_next
            self.X_step = numpy.vstack((self.X_step, x_next))
        return()
    def generate_update_ep_list(self):
        if self.adapt_epsilon=="dual_averaging":
            out = return_update_ep_list(self.tune_l, ini_buffer=75, end_buffer=50, window_size=25)
        elif self.adapt_epsilon=="GPyOpt":
            out = return_update_GPyOpt_list(self.tune_l)
        return(out)

    def generate_update_cov_list(self):
        out = list(range(self.ini_buffer,self.tune_l-self.end_buffer))
        return()
    def generate_update_t_list(self):
        # only called when using GPyOpt
        # can be coupled with epsilon
        out = return_update_GPyOpt_list(tune_l=self.tune_l,window_size=round(self.tune_l/self.num_gpyopt_point))
        return(out)


    def generate_update_delta_list(self):
        # only called when using gpyopt
        out = return_update_GPyOpt_list(tune_l=self.tune_l,window_size=round(self.tune_l/self.num_gpyopt_point))
        return(out)







def thin(experiment):
    # return thinned output

    return()



























