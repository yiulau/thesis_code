from multiprocessing import Pool
import numpy,multiprocessing
from abstract.abstract_adapt_util import welford, adapt_ep, return_update_ep_list
import GPyOpt
from explicit.adapt_util import return_update_GPyOpt_list
import torch
from abstract.abstract_genleapfrog_ult_util import *
from abstract.abstract_leapfrog_ult_util import *
from adapt_util.return_update_list import return_update_lists
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

    def __init__(self,sampler_one_step,adapter=None,num_samples_per_chain=2000,thin=1,warm_up_per_chain=None,initialization=None,num_chains=4,parallel_computing=False,is_float=False,isstore_to_disk=False):
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
                result_sequential.append(one_chain_experiment.run)
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


par_type_dict= {"ep":"fast","L":"medium","t":"medium","alpha":"medium","xhmc_delta":"medium","diag_cov":"slow","cov":"slow"}

# unique for individual sampler
tune_method_dict = {"ep":"opt","t":"opt"}

def generate_sampler_one_step(Ham,windowed,dynamic,second_order,is_float,fixed_tune_dict,tune_dict):

    if is_float==True:
        precision_type = 'torch.FloatTensor'
    else:
        precision_type = 'torch.DoubleTensor'
        #
    torch.set_default_tensor_type(precision_type)
    input_time = False
    out = tuneable_param(dynamic,second_order,Ham.metric.name,Ham.metric.criterion,input_time)
    for param in out:
        # tune_method one of {fixed,dual,opt,adapt}
        tune_method = tune_method_dict[param]
        # par_type one of {fast,medium,slow}
        par_type = par_type_dict[param]
        if tune_method=="fixed":
            pass
        elif tune_method=="dual":
            adapter.add_param_dual(param,par_type)
        elif tune_method=="opt":
            adapter.add_param_opt(param,par_type)
        elif tune_method=="adapt":
            adapter.add_param_adapt(param,par_type)
        else:
            raise ValueError("unknown tune method")
        if par_type=="fast":
            adapter.add_param_fast(param,tune_method)
        elif par_type=="medium":
            adapter.add_param_medium(param,tune_method)
        elif par_type=="slow":
            adapter.add_param_slow(param,tune_method)
        else:
            raise ValueError("unknow par type")


    out = wrap(windowed,fixed_tune_dict,tune_dict)


# want it so that sampler_one_step only has inputq and tuning paramater
# supply tuning parameter with a dictionary
# fixed_tune dict stores name and val of tuning paramter that stays the same throughout the entire chain
# tune dict stores two things : param tuned by pyopt, parm tuned by dual averaging
# always start tuning ep first
#
def wrap(windowed,fixed_tune_dict,tune_dict,tune_par_dict):
    # tune_par_setting = tuple = (Tunable,value,tune_by)
    # Tunable is a boolean variable . True means the variable will be tuned. False if fixed
    # value is the fixed param val if Tunable == False, initial value if Tunable == True
    # tune_by_dual is a boolean variable . True if we are tuning hte variable by dual averaging
    # false if by bayesian optimization
    # epsilon
    #if "epsilon" in fixed_tune_dict:
        #ep_setting = (False,fixed_tune_dict["epsilon"][0],fixed_tune_dict["epsilon"][1])
    #if "epsilon" in fixed_tune_dict:
    #    ep_setting = fixed_tune_dict["epsilon"]
    #else:
     #   ep_setting = tune_dict["epsilon"]
    ep_setting = tune_par_dict["epsilon"]
    t_setting = tune_par_dict["t"]
    delta_setting = tune_par_dict["delta"]
    alpha_setting = tune_par_dict["alpha"]
    if ep_setting[0]==False:
        pass
    else:
        pass


class one_chain_obj(object):
    def __init__(self,Ham,windowed,precision,sampling_metadata=None,initialization=None,adapter=None):
        self.sampling_metadata = sampling_metadata
        self.precision = precision
        self.sampler_one_step = generate_sampler_one_step(Ham,windowed,precision,fixed_tune_dict,tune_dict)
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






def thin(experiment):
    # return thinned output

    return()



























