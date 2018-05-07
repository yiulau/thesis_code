from adapt_util.adapt_util import tuneable_param
from general_util.pytorch_util import welford

import time
class integrator(object):
    def __init__(self,dynamic,second_order,criterion=None,input_time="L"):
        self.dynamic = dynamic
        self.second_order = second_order
        self.criterion = criterion
        self.input_time=input_time
        self.ave_second_per_leapfrog = 0
        if self.dynamic:
            if self.criterion is None:
                raise ValueError("criterion needs to be defined for dynamic integrator")

    def evolve(self):
        start = time.time()
        self.run()
        total_seconds = time.time() - start
        ave_seconds = total_seconds/self.num_transitions
        self.ave_second_per_leapfrog = self.welford_obj.mean(self.ave_second_per_leapfrog,ave_seconds)

    def set_tunable_param(self,metric):
        self.tunable_param = tuneable_param(self.dynamic,self.second_order,metric,self.criterion,self.input_time)

    def find_ave_second_per_leapfrog(self):
        if self.ave_second_per_leapfrog==0:
            self.welford_obj = welford()
            for i in range(20):
                self.evolve()
        return(self.ave_second_per_leapfrog)
