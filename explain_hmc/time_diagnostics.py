import time
class time_diagnositcs(object):
    def __init__(self):
        self.time_one_trans = time.time()
        self.num_grad_one_trans = 0
        self.num_H_eval_one_trans = 0

    def add_num_grad(self,num):
        self.num_grad_one_trans += num

    def add_num_H_eval(self,num):
        self.num_H_eval_one_trans +=num

    def update_time(self):
        self.time_one_trans = time.time() - self.time_one_trans

    def renew(self):
        self.time_one_trans = time.time()
        self.num_grad_one_trans = 0
        self.num_H_eval_oen_trans = 0





#time_object = time_diagnositcs()
#time.sleep(4)
#time_object.add_one_num_grad()
#time_object.add_one_num_H_eval()
#time_object.update_time()

#print(time_object.time_one_trans,time_object.num_grad_one_trans,time_object.num_H_eval_one_trans)