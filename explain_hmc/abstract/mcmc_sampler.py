import time
from multiprocessing import Pool
import numpy,multiprocessing,os,pickle,copy,torch
from adapt_util.tune_param_classes.tune_param_class import tune_param_objs_creator
from adapt_util.tune_param_classes.tuning_param_obj import tuning_param_states
from explicit.adapt_util import return_update_GPyOpt_list
from abstract.abstract_genleapfrog_ult_util import *
from abstract.abstract_leapfrog_ult_util import *
from adapt_util.return_update_list import return_update_lists
from abstract.integrator import sampler_one_step
from adapt_util.adapter_class import adapter_class
from adapt_util.tune_param_classes.tune_param_setting_util import default_adapter_setting
from general_util.memory_util import to_pickle_memory
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

def mcmc_sampler_settings_dict(mcmc_id,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,tune_l_per_chain=5,warmup_per_chain=1000,is_float=False,isstore_to_disk=False,same_init=False):
        # mcmc_id should be a dictionary
        out = {}
        out.update({"num_chains":num_chains,"num_cpu":num_cpu,"thin":thin,"warmup_per_chain":warmup_per_chain})
        out.update({"is_float":is_float,"isstore_to_disk":isstore_to_disk,"mcmc_id":mcmc_id})
        out.update({"num_samples_per_chain":samples_per_chain,"same_init":same_init,"tune_l_per_chain":tune_l_per_chain})
        return(out)
class mcmc_sampler_settings(object):
    def __init__(self,mcmc_id,samples_per_chain=10,num_chains=4,num_cpu=1,thin=1,warmup_per_chain=5,tune_l_per_chain=1000,is_float=False,isstore_to_disk=False,same_init=False):

        # mcmc_id should be a dictionary
        self.num_chains = num_chains
        self.num_cpu = num_cpu
        self.thin = thin
        self.warmup_per_chain = warmup_per_chain
        self.tune_l_per_chain = tune_l_per_chain
        self.is_float = is_float
        self.isstore_to_disk = isstore_to_disk
        self.mcmc_id = mcmc_id
        self.num_samples_per_chain = samples_per_chain
        self.same_init = same_init

class mcmc_sampler(object):
    # may share experiment id with other sampling_objs
    # tune_settings_dict one object for all 4 (num = numchains) chains
    def __init__(self,tune_dict,mcmc_settings_dict,tune_settings_dict,experiment_obj=None,init_q_point_list=None,adapter_setting=None):
        #for name, val in mcmc_meta_obj.__dict__.items():
         #   setattr(self,name,val)
        self.chains_ready = False
        self.tune_dict = tune_dict
        self.tune_settings_dict = tune_settings_dict
        self.mcmc_settings_dict = mcmc_settings_dict
        self.num_chains = self.mcmc_settings_dict["num_chains"]
        self.same_init = self.mcmc_settings_dict["same_init"]
        self.store_chains = numpy.empty(self.num_chains, object)
        if init_q_point_list is None:
            self.init_q_point_list = default_init_q_point_list(v_fun=self.tune_dict["v_fun"],
                                                               num_chains=self.num_chains,
                                                               same_init=self.same_init)
        else:
            self.init_q_point_list= init_q_point_list

        #self.Ham = Ham
        if not experiment_obj is None:
            self.experiment_id = experiment_obj.id
        else:
            self.experiment_id = None
        if adapter_setting is None:
            self.adapter_setting = default_adapter_setting()
        else:
            self.adapter_setting = adapter_setting
        if not hasattr(self,"sampler_id"):
            self.sampler_id = 0


    def prepare_chains(self):
        # same init = True means the chains will have the same initq, does not affect tuning parameters
        #initialization_obj = initialization(same_init)
        self.num_samples_per_chain = self.mcmc_settings_dict["num_samples_per_chain"]
        self.isstore_to_disk = self.mcmc_settings_dict["isstore_to_disk"]
        self.warmup_per_chain = self.mcmc_settings_dict["warmup_per_chain"]
        self.tune_l_per_chain = self.mcmc_settings_dict["tune_l_per_chain"]
        for i in range(self.num_chains):
            #if same_init:
             #   initialization_obj = initialization()
            #print(self.experiment_id)
            #print(self.init_q_point_list)
            #print(self.init_q_point_list[i].flattened_tensor)

            this_chain_setting = one_chain_settings_dict(sampler_id=self.sampler_id,chain_id=i,
                                                     experiment_id=self.experiment_id,
                                                     num_samples=self.num_samples_per_chain,
                                                     warm_up=self.warmup_per_chain,tune_l=self.tune_l_per_chain)
            this_tune_dict = self.tune_dict.copy()
            this_chain_obj = one_chain_obj(sampler_obj=self,init_point=self.init_q_point_list[i],
                                           tune_dict=self.tune_dict,chain_setting=this_chain_setting,
                                           tune_settings_dict=self.tune_settings_dict.copy(),
                                           adapter_setting=self.adapter_setting)
            this_chain_obj.prepare_this_chain()
            this_chain_obj.isstore_to_disk = self.isstore_to_disk
            this_chain_obj.warmup_per_chain = self.warmup_per_chain
            #this_chain_obj.tune_settings_dict = self.tune_settings_dict.copy()
            #print(i)
            self.store_chains[i] = {"chain_obj":this_chain_obj}
        self.chains_ready = True
        #print(self.store_chains)

    def run_chain(self,chain_id):
        if not self.chains_ready:
            self.prepare_chains()
            #raise ValueError("run self.prepare_chains() firstf")
        #print("yes")
        #exit()
        (self.store_chains[chain_id]["chain_obj"]).run()
        output = self.store_chains[chain_id]["chain_obj"].store_samples
        return()
    def start_sampling(self):
        self.num_cpu = self.mcmc_settings_dict["num_cpu"]
        if not self.chains_ready:
            self.prepare_chains()
        if self.num_cpu>1:
            # do something parallel
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                result_parallel = pool.map(self.run_chain, range(self.num_chains))
            #out = result_parallel
        else:
            result_sequential = []
            #print("yes")
            #print(self.num_chains)
            #exit()
            for i in range(self.num_chains):
                #print(i)
                #self.run(0)
                result_sequential.append(self.run_chain(i))
                #experiment = one_chain_sampling(self.precision, self.initialization, self.sampler_one_step, self.adapter)
            #out = result_sequential
        return()

    def pre_sampling_diagnostics(self,test_run_chain_length=15):
        # get
        # 1 estimated total volume
        # 2 estimated  computing time (per chain )
        # 3 estimated total computing time (serial)
        # 4 estimated total coputing time (if parallel, given number of agents)
        # 5 ave number of time per leapfrog
        #initialization_obj = initialization(chain_length = test_run_chain_length)
        #temp_experiment = self.experiment.clone(initialization_obj)
        #temp_sampler =
        #print(self.mcmc_settings_dict)
        #print("yes")
        self.num_samples_per_chain = self.mcmc_settings_dict["num_samples_per_chain"]
        #print(self.store_chains[0]["chain_obj"].adapter.adapter_meta.tune)
        #exit()
        #print(self.store_chains[0])
        #exit()
        if self.store_chains[0]["chain_obj"].adapter.adapter_meta.tune:

            temp_mcmc_meta = self.mcmc_settings_dict.copy()
            temp_mcmc_meta["num_chains"]=1
            temp_mcmc_meta["num_cpu"] = 1
            temp_mcmc_meta["isstore_to_disk"] = False
            temp_mcmc_meta["num_samples_per_chain"] = temp_mcmc_meta["warmup_per_chain"] + test_run_chain_length
        else:
            temp_mcmc_meta = self.mcmc_settings_dict.copy()
            temp_mcmc_meta["num_chains"] = 1
            temp_mcmc_meta["num_cpu"] = 1
            temp_mcmc_meta["isstore_to_disk"] = False
            temp_mcmc_meta["num_samples_per_chain"]=test_run_chain_length

        temp_sampler = mcmc_sampler(tune_dict=self.tune_dict,mcmc_settings_dict=temp_mcmc_meta,
                                    tune_settings_dict=self.tune_settings_dict,
                                    experiment_obj=None, init_q_point_list=None, adapter_setting=self.adapter_setting)


        temp_sampler.start_sampling()
        #exit()
        out = {}
        diagnostics = temp_sampler.sampler_metadata.diagnostics()
        total_warm_up_iter = diagnostics["total_warm_up_iter"]
        total_fixed_tune_iter = diagnostics["total_fixed_tune_iter"]
        total_warm_up_time = diagnostics["total_warm_up_time"]
        total_fixed_tune_time = diagnostics["total_fixed_tune_time"]
        total_num_samples = total_fixed_tune_iter+total_fixed_tune_iter
        total_time = total_warm_up_time+total_fixed_tune_time
        #temp_ave_second_per_leapfrog = temp_output.sample_metadata.ave_second_per_leapfrog
        #temp_num_samples = temp_sampler.sampler_metadta.get_num_samples()
        total_size = temp_sampler.sampler_metadata.get_size_mb()
        estimated_total_volume = total_size * (self.num_chains * self.num_samples_per_chain)/total_num_samples
        out.update({"total_volume":estimated_total_volume})

        estimated_compute_wu_seconds_per_chain = total_warm_up_time * self.warmup_per_chain/total_warm_up_iter
        out.update({"warm up seconds_per_chain":estimated_compute_wu_seconds_per_chain})
        fixed_tune_per_chain = self.num_samples_per_chain - self.warmup_per_chain
        estimated_compute_ft_seconds_per_chain = total_fixed_tune_time * fixed_tune_per_chain / total_warm_up_iter
        out.update({"fixed tune seconds_per_chain": estimated_compute_wu_seconds_per_chain})
        estimated_compute_seconds_per_chain  = estimated_compute_wu_seconds_per_chain + estimated_compute_ft_seconds_per_chain
        estimated_compute_seconds = self.num_chains * estimated_compute_seconds_per_chain
        out.update({"total_seconds":estimated_compute_seconds})
        estimated_compute_seconds_parallel = estimated_compute_seconds/self.num_cpu
        out.update({"total_seconds with parallel":estimated_compute_seconds_parallel})


        return(out)
    def get_samples(self,permuted=True):
        # outputs numpy matrix
        if permuted:
            temp_list = []
            for chain in self.store_chains:
                temp_list.append(chain["chain_obj"].get_samples(warmup=self.warmup_per_chain))
            output = temp_list[0]
            if len(temp_list)>0:
                for i in range(1,len(temp_list)):
                    output = numpy.concatentate([output,temp_list[i]],axis=0)
            return(output)
        else:
            raise ValueError("for now leave this")
# metadata only matters after sampling has started
class sampler_metadata(object):
    def __init___(self,mcmc_sampler_obj):
        self.mcmc_sampler_obj = mcmc_sampler_obj
        self.total_time = 0

    def store_to_disk(self):
        if self.store_address is None:
            self.store_address = "mcmc_sampler.pkl"
        with open(self.store_address, 'wb') as f:
            pickle.dump(self, f)

    def start_time(self):
        self.start_time = time.time()
    def end_time(self):
        self.total_time += self.start_time - time.time()

    def diagnostics(self):
        tune_l_time_list = []
        fixed_tune_time_list = []
        num_tune_l_iter_list = []
        num_fixed_tune_iter_list = []
        for chain in self.mcmc_sampler_obj.store_chains:
            chain_obj = chain["chain_obj"]
            tune_l_time_this_chain_list = [0]*chain_obj.chain_setting["num_samples"]
            fixed_tune_time_this_chain_list = [0]*chain_obj.chain_setting["num_samples"]
            num_tune_l_iter_this_chain = chain_obj.adapter.adapter_setting["tune_l"]
            num_fixed_tune_this_chain = chain_obj.chain_setting["num_samples"] - num_tune_l_iter_this_chain
            for i in range(num_tune_l_iter_this_chain):
                tune_l_time_this_chain_list[i] = chain_obj.stores_samples[i]["log_obj"].time_since_creation
            for i in range(num_tune_l_iter_this_chain, num_tune_l_iter_this_chain + num_fixed_tune_this_chain):
                fixed_tune_time_this_chain_list[i] = chain_obj.stores_samples[i]["log_obj"].time_since_creation

            tune_l_time_list.append(tune_l_time_this_chain_list)
            fixed_tune_time_list.append(fixed_tune_time_this_chain_list)
            num_tune_l_iter_list.append(num_tune_l_iter_this_chain)
            num_fixed_tune_iter_list.append(num_fixed_tune_this_chain)

        total_tune_l_time = 0
        total_fixed_tune_time = 0
        total_tune_l_iter = 0
        total_fixed_tune_iter = 0
        for i in range(self.mcmc_sampler_obj.num_chains):
            total_tune_l_time += sum(tune_l_time_list[i])
            total_fixed_tune_time += sum(fixed_tune_time_list[i])
            total_tune_l_iter += num_tune_l_iter_list[i]
            total_fixed_tune_iter +=num_fixed_tune_iter_list[i]

        out = {"total_tune_l_time":total_tune_l_time,"total_fixed_tune_time":total_fixed_tune_time}
        out.update({"total_tune_l_iter":total_tune_l_iter,"total_fixed_tune_iter":total_fixed_tune_iter})
        out.update({"tune_l_time_list":tune_l_time_list,"fixed_tune_time_list":fixed_tune_time_list})
        return(out)
    def get_num_samples(self):
        sum = 0
        for i in range(self.mcmc_sampler_obj.num_chains):
            sum+= len(self.mcmc_sampler_obj.store_chains[i].stores_samples)
        return(sum)

    def get_size_mb(self):
        # save to disk. measure volume, then remove stored copy
        # with open("temp_sampler_volume.pkl", 'wb') as f:
        #     pickle.dump(self.mcmc_sampler_obj, f)
        # size = os.path.getsize("./temp_sampler_volume.pkl") / (1024. * 1024)
        # os.remove("./temp_sampler_volume.pkl")
        size = to_pickle_memory(self)
        return(size)




#par_type_dict= {"epsilon":"fast","evolve_L":"medium","evolve_t":"medium","alpha":"medium","xhmc_delta":"medium","diag_cov":"slow","cov":"slow"}

# unique for individual sampler
#tune_method_dict = {"epsilon":"opt","evolve_t":"opt"}


# want it so that sampler_one_step only has inputq and tuning paramater
# supply tuning parameter with a dictionary
# fixed_tune dict stores name and val of tuning paramter that stays the same throughout the entire chain
# tune dict stores two things : param tuned by pyopt, parm tuned by dual averaging
# always start tuning ep first

# tune_dict for each chain should be independent
class one_chain_obj(object):
    def __init__(self,sampler_obj,init_point,tune_dict,chain_setting,
                 tune_settings_dict,adapter_setting=None):
        self.chain_setting = chain_setting
        self.store_samples = []
        self.chain_ready = False
        self.tune_settings_dict = tune_settings_dict



        #print(self.sampling_metadata.__dict__)
        #for param,val in self.sampling_metadata.__dict__.items():
        #    setattr(self,param,val)
        self.tune_dict = tune_dict
        #if tuning_param_settings is None:
        #    self.tuning_param_settings = tuning_param_settings(tune_dict)
        #else:
        #    self.tuning_param_settings = tuning_param_settings
        #print(adapter_setting is None)
        #exit()
        if adapter_setting is None:
            self.adapter = adapter_class(one_chain_obj=self)
        else:
            self.adapter = adapter_class(one_chain_obj=self,adapter_setting=adapter_setting)


        self.tune_param_objs_dict = tune_param_objs_creator(tune_dict=tune_dict,adapter_obj=self.adapter,
                                                           tune_settings_dict=tune_settings_dict)


        #print(self.tune_param_objs_dict)
        #exit()
        self.tuning_param_states = tuning_param_states(adapter=self.adapter,param_objs_dict=self.tune_param_objs_dict)
        self.adapter.tuning_param_states = self.tuning_param_states
        #print(self.tuning_param_states)
        #exit()
        #self.adapter.tuning_param_states = self.tuning_param_states
        #if initialization_obj.tune_param_obj_dict is None:
        #    self.tune_param_obj_dict = tune_params_obj_creator(tune_dict,self.tuning_param_settings)
        #else:
        #    self.tune_param_obj_dict = initialization_obj.tune_param_obj_dict
        #self.sampler_one_step = sampler_one_step(self.tune_dict)
        #print(self.tune_param_objs_dict["epsilon"].get_val())
        #print(self.tune_param_objs_dict["evolve_L"].get_val())
        #exit()
        #self.sampler_one_step = sampler_one_step(self.tune_param_objs_dict,init_point)
        self.sampler_one_step = sampler_one_step(tune_param_objs_dict=self.tune_param_objs_dict,init_point=init_point,
                                                 tune_dict=tune_dict)
        if "epsilon" in self.tune_param_objs_dict:
            ep_obj = self.tune_param_objs_dict["epsilon"]
            ep_obj.Ham = self.sampler_one_step.Ham

        #exit()
    def adapt(self,sample_obj):
        # if self.adapter is none, do nothing
        self.adapter.update(sample_obj)
        #self.sampler_one_step = out.sampler_one_step

    def store_to_disk(self):
        if self.store_address is None:
            self.store_address = "chain.mkl"
        with open(self.store_address, 'wb') as f:
            pickle.dump(self, f)

    def prepare_this_chain(self):
        # initiate tuning parameters if they are tuned automatically
        self.log_obj = log_class()
        self.sampler_one_step.log_obj = self.log_obj
        temp_dict = {}
        for name,obj in self.tune_param_objs_dict.items():
            priority = obj.update_priority
            temp_dict.update({priority:obj})
        temp_tuple = sorted(temp_dict.items())
        for priority,obj in temp_tuple:
                obj.initialize_tuning_param()
        self.chain_ready = True

    def run(self):
        #self.initialization_obj.initialize()
        #print(self.warmup_per_chain)

        if not self.chain_ready:
            raise ValueError("need to run prepare this chain")
        temp = self.chain_setting["thin"]
        #print("yes")
        #exit()
        for counter in range(self.chain_setting["num_samples"]):
        #for counter in range(5):
            temp -= 1
            if not temp>0.1:
                keep = True
                cur = self.chain_setting["thin"]
            else:
                keep = False

            print(self.tune_param_objs_dict["epsilon"].get_val())

            out = self.sampler_one_step.run()
            #self.adapter.log_obj = self.log_obj
            sample_dict = {"q":out,"iter":counter,"log":self.log_obj.snapshot()}
            if keep:
                self.add_sample(sample_dict=sample_dict)
                if self.is_to_disk_now(counter):
                    self.store_to_disk()
            if counter < self.chain_setting["tune_l"]:
                out.iter = counter
                self.adapt(sample_dict)
            #print("tune_l is {}".format(self.chain_setting["tune_l"]))
            print(out.flattened_tensor)
            print("iter is {}".format(counter))
            print("epsilon val {}".format(self.tune_param_objs_dict["epsilon"].get_val()))
#            print("evolve_L val {}".format(self.tune_param_objs_dict["evolve_L"].get_val()))
            print("accept_rate {}".format(self.log_obj.store["accept_rate"]))
            print("divergent is {}".format(self.log_obj.store["divergent"]))

        return()
    def add_sample(self,sample_dict):
        #print(self.log_obj.store)
        self.store_samples.append(sample_dict)

    def is_to_disk_now(self,counter):
        return(False)

    def get_samples(self,warmup=None):
        if warmup is None:
            warmup = self.chain_setting["warmup"]
        num_out = len(self.store_samples) - warmup
        assert num_out >=1
        store_torch_matrix = torch.zeros(num_out,len(self.store_samples[0]["q"].flattened_tensor))
        # load into tensor matrix
        for i in range(num_out):
            store_torch_matrix[i,:].copy_(self.store_samples[i+warmup]["q"].flattened_tensor)

        store_matrix = store_torch_matrix.numpy()
        return(store_matrix)

# log_obj should keep information about dual_obj, and information about tuning parameters
# created at the start of each transition. discarded at the end of each transition
class log_class(object):
    def __init__(self):
        self.store = {}
        self.start_time = time.time()
    def snapshot(self):
        self.time_since_creation = self.get_time_since_creation()
        return(self)

    def get_time_since_creation(self):
        return(time.time()-self.start_time)




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

# class one_chain_settings(object):
#     def __init__(self,sampler_id,chain_id,num_samples=10,thin=1,experiment_id=None,tune_l=5):
#         # one for every chain. in everything sampling object, in every experiment
#         # parallel chains sampling from the same distribution shares sampling_obj
#         # e.g. 4 chains sampling from iid normal
#         # different chains sampling in the context of the same experiment shares experiment_obj
#         # e.g. HMC(ep,t) sampling from a model with different values of (ep,t) on a grid
#         # thin an integer >=0 skips
#         # period for saving samples
#         self.experiment_id = experiment_id
#         self.sampler_id = sampler_id
#         self.chain_id = chain_id
#         self.num_samples = num_samples
#         self.thin = thin
#         self.tune_l = tune_l
#
#     #def clone(self):


def one_chain_settings_dict(sampler_id,chain_id,num_samples=10,thin=1,experiment_id=None,tune_l=5,warm_up=5):

        # one for every chain. in everything sampling object, in every experiment
        # parallel chains sampling from the same distribution shares sampling_obj
        # e.g. 4 chains sampling from iid normal
        # different chains sampling in the context of the same experiment shares experiment_obj
        # e.g. HMC(ep,t) sampling from a model with different values of (ep,t) on a grid
        # thin an integer >=0 skips
        # period for saving samples
        out = {"experiment_id":experiment_id,"sampler_id":sampler_id,"chain_id":chain_id}
        out.update({"num_samples":num_samples,"thin":thin,"tune_l":tune_l,"warm_up":warm_up})

        return(out)
























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



def default_init_q_point_list(v_fun,num_chains,same_init=False):
    v_obj = v_fun()
    init_q_point_list = [None]*num_chains
    if same_init:
        #print("yes")
        temp_point = v_obj.q_point.point_clone()
        temp_point.flattened_tensor.copy_(torch.randn(len(temp_point.flattened_tensor)))
        temp_point.load_flatten()
        #print(temp_point.flattened_tensor)
        #print(temp_point)
        #exit()
        for i in range(num_chains):
            init_q_point_list[i] = temp_point.point_clone()
            #print(init_q_point_list[i].flattened_tensor)
    else:
        for i in range(num_chains):
            temp_point = v_obj.q_point.point_clone()
            temp_point.flattened_tensor.copy_(torch.randn(len(temp_point.flattened_tensor)))
            temp_point.load_flatten()
            init_q_point_list[i] = temp_point



    return(init_q_point_list)
