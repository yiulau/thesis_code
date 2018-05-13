import time
from multiprocessing import Pool
import numpy,multiprocessing,os,pickle,copy,torch

from explicit.adapt_util import return_update_GPyOpt_list
from abstract.abstract_genleapfrog_ult_util import *
from abstract.abstract_leapfrog_ult_util import *
from adapt_util.return_update_list import return_update_lists
from abstract.integrator import sampler_one_step
from adapt_util.adapter_class import adapter_class

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

class mcmc_sampler_settings(object):
    def __init__(self,mcmc_id,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,warmup_per_chain=5,is_float=False,isstore_to_disk=False):
        self.mcmc_id = mcmc_id
        # mcmc_id should be a dictionary
        self.num_chains = num_chains
        self.num_cpu = num_cpu
        self.thin = thin
        self.warmup_per_chain = warmup_per_chain
        self.is_float = is_float
        self.isstore_to_disk = isstore_to_disk
        self.mcmc_id = mcmc_id
        self.num_samples_per_chain = samples_per_chain

class mcmc_sampler(object):
    # may share experiment id with other sampling_objs
    # tune_settings_dict one object for all 4 (num = numchains) chains
    def __init__(self,tune_dict,mcmc_meta_obj,init_q_list=None,tune_settings_dict=None,experiment_obj=None):
        for name, val in mcmc_meta_obj.__dict__.items():
            setattr(self,name,val)
        self.store_chains = numpy.empty(self.num_chains, object)
        self.tune_dict = tune_dict
        if init_q_list is None:
            self.init_q_list = [None]*self.num_chains
        else:
            self.init_q_list = init_q_list
        self.tune_settings_dict = tune_settings_dict
        #self.Ham = Ham
        if not experiment_obj is None:
            self.experiment_id = experiment_obj.id
        else:
            self.experiment_id = None
        self.sampling_id = 0
        self.chains_ready = False
    def prepare_chains(self,same_init=False):
        # same init = True means the chains will have the same initq, does not affect tuning parameters
        #initialization_obj = initialization()

        for i in range(self.num_chains):
            #if same_init:
             #   initialization_obj = initialization()
            #print(self.experiment_id)

            this_chain_metadata = one_chain_settings(self.sampling_id,chain_id=i,experiment_obj=self.experiment_id,
                                                     num_samples=self.num_samples_per_chain)
            this_tune_dict = self.tune_dict.copy()
            this_chain_obj = one_chain_obj(self,this_tune_dict,this_chain_metadata,
                                           self.init_q_list[i],self.tune_settings_dict.copy())
            this_chain_obj.isstore_to_disk = self.isstore_to_disk
            this_chain_obj.warmup_per_chain = self.warmup_per_chain
            #this_chain_obj.tune_settings_dict = self.tune_settings_dict.copy()
            self.store_chains[i] = {"chain_obj":this_chain_obj}
            self.chains_ready = True
    def run(self,chain_id):
        if not self.chains_ready:
            raise ValueError("run self.prepare_chains() firstf")
        (self.store_chains[chain_id]["chain_obj"]).run()
        output = self.store_chains[chain_id]["chain_obj"].store_samples
        return(output)
    def start_sampling(self):
        if not self.chains_ready:
            self.prepare_chains()
        if self.num_cpu>1:
            # do something parallel
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                result_parallel = pool.map(self.run, range(self.num_chains))

        else:
            result_sequential = [None]*self.num_chains
            for i in range(self.num_chains):
                #print(i)
                #self.run(0)
                result_sequential.append(self.run(i))
                #experiment = one_chain_sampling(self.precision, self.initialization, self.sampler_one_step, self.adapter)

        return(result_sequential)

    def pre_experiment_diagnostics(self,test_run_chain_length=15):
        # get
        # 1 estimated total volume
        # 2 estimated  computing time (per chain )
        # 3 estimated total computing time (serial)
        # 4 estimated total coputing time (if parallel, given number of agents)
        # 5 ave number of time per leapfrog
        initialization_obj = initialization(chain_length = test_run_chain_length)
        temp_experiment = self.experiment.clone(initialization_obj)

        temp_output = temp_experiment.run()
        out = {}
        time = temp_output.sampling_metadata.total_time
        ave_per_leapfrog = temp_output.sample_metadata.ave_second_per_leapfrog
        total_size = temp_output.sample_metadata.size_mb
        estimated_total_volume = total_size * (self.num_chains * self.num_per_chain)/test_run_chain_length
        out.append({"total_volume":estimated_total_volume})
        estimated_compute_seconds_per_chain = time * self.num_per_chain/test_run_chain_length
        out.append({"seconds_per_chain":estimated_compute_seconds_per_chain})
        estimated_compute_seconds = self.num_chains * estimated_compute_seconds_per_chain
        out.append({"total_seconds":estimated_compute_seconds})
        estimated_compute_seconds_parallel = estimated_compute_seconds/self.num_agents
        out.append({"total_seconds with parallel":estimated_compute_seconds_parallel})
        with open('model.pkl', 'wb') as f:
            pickle.dump(temp_output, f)
        size = os.path.getsize("./model.pkl") / (1024. * 1024)
        os.remove("./model.pkl")

        return(out)

#par_type_dict= {"epsilon":"fast","evolve_L":"medium","evolve_t":"medium","alpha":"medium","xhmc_delta":"medium","diag_cov":"slow","cov":"slow"}

# unique for individual sampler
#tune_method_dict = {"epsilon":"opt","evolve_t":"opt"}


# want it so that sampler_one_step only has inputq and tuning paramater
# supply tuning parameter with a dictionary
# fixed_tune dict stores name and val of tuning paramter that stays the same throughout the entire chain
# tune dict stores two things : param tuned by pyopt, parm tuned by dual averaging
# always start tuning ep first
from adapt_util.tune_param_classes.tune_param_class import tune_params_obj_creator
from adapt_util.tune_param_classes.tuning_param_obj import tuning_param_states
# tune_dict for each chain should be independent
class one_chain_obj(object):
    def __init__(self,sampling_obj,tune_dict,sampling_metaobj=None,initialization_obj=None,
                 tune_settings_dict=None):
        self.sampling_metadata = sampling_metaobj
        self.store_samples = []
        self.chain_ready = False
        if not hasattr(self,"sampling_metadata"):
            if not self.sampling_metadata is None:
                self.sampling_metadata = one_chain_settings(self.sampling_id,0,experiment_obj=self.experiment_id)

        #print(self.sampling_metadata.__dict__)
        for param,val in self.sampling_metadata.__dict__.items():
            setattr(self,param,val)
        self.tune_dict = tune_dict
        #if tuning_param_settings is None:
        #    self.tuning_param_settings = tuning_param_settings(tune_dict)
        #else:
        #    self.tuning_param_settings = tuning_param_settings
        #if adapter is None:
        self.adapter = adapter_class(self)
        self.tuning_param_states = tuning_param_states(self.adapter)
        self.tune_param_obj_dict = tune_params_obj_creator(tune_dict,tune_settings_dict,self.adapter)

        #if initialization_obj.tune_param_obj_dict is None:
        #    self.tune_param_obj_dict = tune_params_obj_creator(tune_dict,self.tuning_param_settings)
        #else:
        #    self.tune_param_obj_dict = initialization_obj.tune_param_obj_dict
        #self.sampler_one_step = sampler_one_step(self.tune_dict)

        self.sampler_one_step = sampler_one_step(self.tune_param_obj_dict)


    def adapt(self,out,counter):
        # if self.adapter is none, do nothing
        if self.adapter==None:
            pass
        else:
            out = self.adapter.update(out,counter)
            self.sampler_one_step = out.sampler_one_step

    def store_to_disk(self):
        if self.store_address is None:
            self.store_address = "chain.mkl"

        with open(self.store_address, 'wb') as f:
            pickle.dump(self.store_samples, f)

    def prepare_this_chain(self):
        # initiate tuning parameters if they are tuned automatically
        self.log_obj = log_class()
        self.sampler_one_step.log_obj = self.log_obj
        temp_dict = {}
        for name,obj in self.tune_param_obj_dict.items():
            priority = obj.update_priority
            temp_dict.update({priority:obj})
        temp_tuple = sorted(temp_dict.items())
        for obj in temp_tuple:
                obj.initialize_tuning_param()


    def run(self):
        #self.initialization_obj.initialize()
        #print(self.warmup_per_chain)

        if not self.chain_ready:
            raise ValueError("need to run prepare this chain")
        temp = self.thin
        for counter in range(self.num_samples):
            temp -= 1
            if not temp>0.1:
                keep = True
                cur = self.thin
            else:
                keep = False


            out = self.sampler_one_step.run()
            #self.adapter.log_obj = self.log_obj
            if keep:
                self.addsample(out.flattened_tensor.clone(), counter)
                if self.is_to_disk_now(counter):
                    self.store_to_disk()
            if counter < self.warmup_per_chain:
                self.adapt(out, counter)


        return()
    def addsample(self,out,counter):
        #print(self.log_obj.store)
        self.store_samples.append({"q":out,"iter":counter,"log":self.log_obj.snapshot()})

    def is_to_disk_now(self,counter):
        return(False)

# log_obj should keep information about dual_obj, and information about tuning parameters
class log_class(object):
    def __init__(self):
        self.store = {}
    def snapshot(self):
        return(self.store.copy())




class initialization(object):
    # should contain a point object
    def __init__(self,V_obj,q_point=None):
        if not q_point is None:
            V_obj.load_point(q_point)
        else:
            self.initialize()

    def initialize(self):
        self.V_obj.q.flattened_tensor.copy_(torch.randn(len(self.V_obj.q.flattened_tensor))*1.41)
        self.V_obj.q.load_flatten()
        return()

class one_chain_settings(object):
    def __init__(self,sampling_obj,chain_id,num_samples=10,thin=1,experiment_obj=None,tune_l=5):
        # one for every chain. in everything sampling object, in every experiment
        # parallel chains sampling from the same distribution shares sampling_obj
        # e.g. 4 chains sampling from iid normal
        # different chains sampling in the context of the same experiment shares experiment_obj
        # e.g. HMC(ep,t) sampling from a model with different values of (ep,t) on a grid
        # thin an integer >=0 skips
        # period for saving samples
        self.experiment_obj = experiment_obj
        self.sampling_obj = sampling_obj
        self.chain_id = chain_id
        self.num_samples = num_samples
        self.thin = thin
        self.tune_l = tune_l

    #def clone(self):




























#
# class mcmc_sampler(object):
#     # may share experiment id with other sampling_objs
#     def __init__(self,Ham,adapter=None,num_samples_per_chain=2000,thin=1,warm_up_per_chain=None,initialization=None,num_chains=4,num_cpu=1,is_float=False,isstore_to_disk=False):
#         self.store_chains = numpy.empty(num_chains, object)
#         self.num_cpu = num_cpu
#         self.num_chains = num_chains
#         self.Ham = Ham
#         if not experiment_obj is None:
#             self.experiment_id = experiment_obj.id
#         else:
#             self.experiment_id = None
#         self.sampling_id = 0
#
#         return()
#
#     def prepare_chains(self,same_init=False):
#         # same init = True means the chains will have the same initq, does not affect tuning parameters
#         for i in range(self.num_chains):
#             if same_init:
#                 initialization_obj = self.initialization_obj.clone()
#             this_chain_metadata = sampling_metadata(self.experiment_id,self.sampling_id,i)
#             this_chain_obj = one_chain_obj(this_chain_metadata,initialization_obj)
#             self.store_chains[i] = {"chain_obj":this_chain_obj}
#
#     def run(self,chain_id):
#         (self.store_chains[chain_id]["chain_obj"]).run()
#         output = self.store_chains[chain_id]["chain_obj"].store_samples
#         return(output)
#     def start_sampling(self):
#         if self.num_cpu>1:
#             # do something parallel
#             with multiprocessing.Pool(processes=self.num_cpu) as pool:
#                 result_parallel = pool.map(self.run, range(self.num_chains))
#
#         else:
#             result_sequential = [None]*self.num_chains
#             for i in range(self.num_chains):
#                 result_sequential.append(self.run(i))
#                 #experiment = one_chain_sampling(self.precision, self.initialization, self.sampler_one_step, self.adapter)
#
#         return()
#
#     def pre_experiment_diagnostics(self,test_run_chain_length=15):
#         # get
#         # 1 estimated total volume
#         # 2 estimated  computing time (per chain )
#         # 3 estimated total computing time (serial)
#         # 4 estimated total coputing time (if parallel, given number of agents)
#         # 5 ave number of time per leapfrog
#         initialization_obj = initialization(chain_length = test_run_chain_length)
#         temp_experiment = self.experiment.clone(initialization_obj)
#
#         temp_output = temp_experiment.run()
#         out = {}
#         time = temp_output.sampling_metadata.total_time
#         ave_per_leapfrog = temp_output.sample_metadata.ave_second_per_leapfrog
#         total_size = temp_output.sample_metadata.size_mb
#         estimated_total_volume = total_size * (self.num_chains * self.num_per_chain)/test_run_chain_length
#         out.append({"total_volume":estimated_total_volume})
#         estimated_compute_seconds_per_chain = time * self.num_per_chain/test_run_chain_length
#         out.append({"seconds_per_chain":estimated_compute_seconds_per_chain})
#         estimated_compute_seconds = self.num_chains * estimated_compute_seconds_per_chain
#         out.append({"total_seconds":estimated_compute_seconds})
#         estimated_compute_seconds_parallel = estimated_compute_seconds/self.num_agents
#         out.append({"total_seconds with parallel":estimated_compute_seconds_parallel})
#         with open('model.pkl', 'wb') as f:
#             pickle.dump(temp_output, f)
#         size = os.path.getsize("./model.pkl") / (1024. * 1024)
#         os.remove("./model.pkl")
#
#         return(out)
#
#


