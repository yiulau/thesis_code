from adapt_util.adapt_util import tuneable_param
from general_util.pytorch_util import welford
from abstract.abstract_static_sampler import *
from abstract.abstract_nuts_util import *
from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
import time
# this object is assumed to be initiated properly
class sampler_one_step(object):
    def __init__(self,tune_dict,tune_param_obj_dict):
        #print(input_obj.input_dict)

        for param_name, val in tune_dict.items():
            setattr(self,param_name,val)
        self.v_obj = self.v_fun()
        if hasattr(tune_dict,"alpha"):
            self.metric = metric(self.metric_name,self.v_obj,tune_dict.alpha)
        else:
            self.metric = metric(self.metric_name,self.v_obj)
        self.Ham = Hamiltonian(self.v_obj,self.metric)
        if not self.dynamic:
            if hasattr(tune_dict,"evolve_t"):
                self.input_time=True
            else:
                self.input_time=False
        self.ave_second_per_leapfrog = 0
        self.one_step_function,self.tuneable_param = self.generate_sampler_one_step(self.windowed,self.dynamic,self.second_order,self.metric_name)
        # self.tuneable_param supplies the names of tuneable parameters that self.one_step_function needs
        self.tune_param_obj_dict = tune_param_obj_dict
        #self.tuneable_param_obj_dict = {}
        #for name in self.tuneable_param:
        #    self.tuneable_param_obj_dict.update({name:tune_param_obj_dict})
        #for i in range(len(self.tuneable_param)):
        #    self.tuneable_param_dict.update({self.tuneable_param[i]:getattr(self,self.tuneable_param[i])})

        self.one_step_function = wrap(self.one_step_function)
    def evolve(self):
        start = time.time()
        self.run()
        total_seconds = time.time() - start
        ave_seconds = total_seconds/self.num_transitions
        self.ave_second_per_leapfrog = self.welford_obj.mean(self.ave_second_per_leapfrog,ave_seconds)

    def set_tunable_param(self,metric):
        self.tuneable_param = tuneable_param(self.dynamic,self.second_order,metric,self.criterion,self.input_time)

    def find_ave_second_per_leapfrog(self):
        if self.ave_second_per_leapfrog==0:
            self.welford_obj = welford()
            for i in range(20):
                self.evolve()
        return(self.ave_second_per_leapfrog)

    def run(self):
        if hasattr(self,"log_obj"):
            #print("yes")
            out = self.one_step_function(self.Ham.V.q_point,self.Ham,self.tuneable_param_dict,self.log_obj)
        else:
            out = self.one_step_function(self.Ham.V.q_point, self.Ham, self.tuneable_param_dict)
        self.Ham.V.q_point = out[0]
        return(self.Ham.V.q_point.point_clone())



    def generate_sampler_one_step(self,windowed, dynamic, second_order, criterion):
        if second_order:
            if dynamic:
                if criterion == "nuts":
                    out = abstract_NUTS
                elif criterion == "gnuts":
                    out = abstract_GNUTS

                elif criterion == "xhmc":
                    out = abstract_NUTS_xhmc
            else:
                if windowed:
                    out = abstract_static_windowed_one_step
                else:
                    out = abstract_static_one_step
        else:
            if windowed:
                out = abstract_static_windowed_one_step
            else:
                out = abstract_static_one_step

        tuneable_par = tuneable_param(dynamic, second_order, self.metric_name, self.criterion, self.input_time)


        return (out,tuneable_par)


def wrap(raw_sampler_one_step):
    # want output to be function so that takes input point object and tune_param_dict
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
    def sampler_one_step(input_point_obj,Ham_obj,tune_param_obj_dict,log_obj=None):
        tune_param_dict = {}
        for param_name,obj in tune_param_obj_dict.items():
            tune_param_dict.update({param_name:obj.get_val()})
        if not log_obj is None:
            tune_param_dict.update({"log_obj":log_obj})
        tune_param_dict.update({"init_q":input_point_obj})
        tune_param_dict.update({"Ham":Ham_obj})
        return(raw_sampler_one_step(**tune_param_dict))

    return(sampler_one_step)



#def generate_sampler_one_step(Ham,windowed,dynamic,second_order,is_float,fixed_tune_dict,tune_dict):

 #   if is_float==True:
        #precision_type = 'torch.FloatTensor'
  #  else:
        #precision_type = 'torch.DoubleTensor'
        #
   # torch.set_default_tensor_type(precision_type)
    #input_time = False



    #out = wrap(windowed,fixed_tune_dict,tune_dict)