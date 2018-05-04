import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import numpy
from general_util.pytorch_util import get_list_stats
from abstract.abstract_class_point import point


# if need to define explicit gradient do it

class V(nn.Module):
    #def __init__(self,explicit_gradient):
    def __init__(self):
        super(V, self).__init__()
        self.V_setup()
        if self.explicit_gradient is None:
            raise ValueError("self.explicit_gradient need to be defined in V_setup")
        if self.need_higherorderderiv is None:
            raise ValueError("self.need_higherorderderiv need to be defined in V_setup")

        #################################################################################
        self.decides_if_flattened()
        self.V_higherorder_setup()
        self.q_point = point(V=self)
        self.diagnostics = None
    #@abc.abstractmethod
    #def V_setup(self):
        ### initialize function parameters
    #    return()
    #@abc.abstractmethod
    #def forward(self):
    #    return()
    #@abc.abstractmethod
    #def load_explicit_gradient(self):
    #    raise NotImplementedError()
    #    return()
    def evaluate_scalar(self):
        # return float or double
        return(self.forward().data[0])
    def gradient(self):
        # load gradients to list(self.parameters()).grad
        if not self.explicit_gradient:
            o = self.forward()
            o.backward()
        else:
            self.load_explicit_gradient()
    def getdV(self,q=None):
        # return list of pytorch variables containing the gradient
        if not q is None:
            self.load_point(q)
        if not self.need_flatten:
            g = grad(self.forward(),self.list_var,create_graph=True)[0]
        else:
            g = grad(self.forward(), self.list_var, create_graph=True)
        self.load_gradient(g)
        if not self.diagnostics is None:
            self.diagnostics.add_num_grad(1)
        return(g)

    def getdV_tensor(self,q=None):
        # return list of pytorch variables containing the gradient
        #if not q.V is self:
        #    raise ValueError("not the same V function object")
        if not q is None:
            self.load_point(q)
        if self.explicit_gradient:
            self.gradient_tensor.copy_(self.load_explicit_gradient())
        else:
            g = grad(self.forward(), self.list_var, create_graph=True)
            self.load_gradient(g)
            if not self.diagnostics is None:
                self.diagnostics.add_num_grad(1)
        return(self.gradient_tensor)

    def getH(self,q=None):
        if not q is None:
            self.load_point(q)
        if not self.need_flatten:
            g = self.getdV()
            H = Variable(torch.zeros(self.dim, self.dim))
            for i in range(self.dim):
                H[i, :] = grad(g[i], self.list_var, create_graph=True)[0]
        # output: H - Pytorch Variable
        # repetitive. only need to compute the upper or lower triangular part than flip
        else:
            g = self.getdV()
            H = numpy.empty((self.num_var,self.num_var),dtype=numpy.ndarray)
            #H[i,j] = dH/dvar_i dvar_j . a pytorch variable with the shape of var_j
            for i in range(self.num_var):
                for j in range(self.num_var):
                    H[i,j] = self.block_2nd_deriv(g[i],self.list_var[j],True,True)

        self.load_gradient(g)
        self.load_Hessian(H)
        return (g,H)
    # reimplement if need to get exact
    def getH_tensor(self,q=None):
        if not q is None:
            self.load_point(q)
        if self.explicit_gradient:
            self.gradient_tensor.copy_(self.load_explicit_gradient())
            self.Hessian_tensor.copy_(self.load_explicit_H())
        else:
            if not self.need_flatten:
                g = self.getdV()
                H = Variable(torch.zeros(self.dim, self.dim))
                for i in range(self.dim):
                    H[i, :] = grad(g[i], self.list_var, create_graph=True)[0]

            # output: H - Pytorch Variable
            # repetitive. only need to compute the upper or lower triangular part than flip
            else:
                g = self.getdV()
                H = numpy.empty((self.num_var,self.num_var),dtype=numpy.ndarray)
                #H[i,j] = dH/dvar_i dvar_j . a pytorch variable with the shape of var_j
                for i in range(self.num_var):
                    for j in range(self.num_var):
                        H[i,j] = self.block_2nd_deriv(g[i],self.list_var[j],True,True)
            self.load_gradient(g)
            self.load_Hessian(H)

        return (self.gradient_tensor,self.Hessian_tensor)
    def getdiagH_tensor(self,q=None):
        if not q is None:
            self.load_point(q)

        #assert self.metric.name =="softabs_diag"
        if not self.explicit_gradient:
            _,H = self.getH_tensor()
            self.diagH_tensor.copy_(torch.diag(H))
        else:
            self.gradient_tensor.copy_(self.load_gradient())
            self.gradient_tensor.copy_(self.load_mdiagH())
        return(self.gradient_tensor,self.diagH_tensor)


    def getdH(self,q=None):
        if not q is None:
            self.load_point(q)
        if not self.need_flatten:
            g, H = self.getH()
            dH = torch.zeros(self.dim, self.dim, self.dim)
            for i in range(self.dim):
                for j in range(self.dim):
                    dH[:, i, j] = grad(H[i, j], self.list_var, create_graph=False, retain_graph=True)[0].data
        else:
            g,H = self.getH()
            dH = numpy.empty((self.num_var,self.num_var,self.num_var),dtype=numpy.ndarray)
            # also repetitive: each of the 3!= 6 permutation computes the same thing
            for i in range(self.num_var):
                for j in range(self.num_var):
                    for k in range(self.num_var):
                        dH[i,j,k] = self.block_3rd_deriv(H[i,j],self.list_var[k],True,True)
        self.load_dH(dH)
        return(g,H,dH)
    def getdH_tensor(self,q=None):
        # takes anything but outputs tensor
        if not q is None:
            self.load_point(q)
        if self.explicit_gradient:
            self.gradient_tensor.copy_(self.load_explicit_gradient())
            self.Hessian_tensor.copy_(self.load_explicit_H())
            self.dH_tensor.copy_(self.load_explicit_dH())
        else:
            if not self.need_flatten:
                g, H = self.getH()
                dH = torch.zeros(self.dim, self.dim, self.dim)
                for i in range(self.dim):
                    for j in range(self.dim):
                        dH[:, i, j] = grad(H[i, j], self.list_var, create_graph=False, retain_graph=True)[0].data
            else:
                g,H = self.getH()
                dH = numpy.empty((self.num_var,self.num_var,self.num_var),dtype=numpy.ndarray)
                # also repetitive: each of the 3!= 6 permutation computes the same thing
                for i in range(self.num_var):
                    for j in range(self.num_var):
                        for k in range(self.num_var):
                            dH[i,j,k] = self.block_3rd_deriv(H[i,j],self.list_var[k],True,True)
            self.load_gradient(g)
            self.load_Hessian(H)
            self.load_dH(dH)

        return(self.gradient_tensor,self.Hessian_tensor,self.dH_tensor)


    def get_graddiagH(self,q=None):
        #returns (dV,mdiagH,mgraddiagH)
        if not q is None:
            self.load_point(q)
        assert self.explicit_gradient == True
        if not self.explicit_gradient:
            _,H,dH = self.getdH_tensor()
            self.diagH_tensor.copy_(torch.diag(H))
            out = torch.zeros(self.dim, self.dim)
            for i in range(self.dim):
                out[i, :] = torch.diag(dH[i, :, :])
            self.graddiagH_tensor.copy_(out)
        else:
            self.gradient_tensor.copy_(self.load_gradient())
            self.gradient_tensor.copy_(self.load_mdiagH())
        #assert self.metric.name == "softabs_diag"
        self.gradient_tensor.copy_(self.load_explicit_gradient())
        self.diagH_tensor.copy_(self.load_explicit_diagH())
        self.graddiagH_tensor.copy_(self.load_explicit_graddiagH())
        return(self.gradient_tensor,self.diagH_tensor,self.graddiagH_tensor)

    def block_2nd_deriv(self,var1,var2,retain_graph,create_graph):
        # return dH/dvar1dvar2 in a multivariate numpy array with the shape of var1
        # where each entry is a pytorch Variable with the shape of var2
        var1_shape_container = numpy.empty(list(var1.shape),dtype=Variable)
        for index, x in numpy.ndenumerate(var1_shape_container):
            var1_shape_container[index] = grad(var1[index],var2,retain_graph=retain_graph,create_graph=create_graph)[0]
        return(var1_shape_container)
    def block_3rd_deriv(self,var1,var2,retain_graph,create_graph):
        # return dH/dvar1dvar2dvar3 in a numpy array in the shape of var1
        # where each entry (dH/dvar1dvar2dvar3[index]) is a numpy array with the shape of var2
        # where each entry (dH/dvar1dvar2dvar3[index])[i] is a Variable with the shape of var 3
        var1_shape_container = numpy.empty(list(var1.shape), dtype=numpy.ndarray)
        for index,x in numpy.ndenumerate(var1_shape_container):
            var1_shape_container[index] = self.block_2nd_deriv(var1[index],var2,retain_graph,create_graph)
        return(var1_shape_container)
    def load_gradient(self,list_g):
        if not self.need_flatten:
            self.gradient_tensor.copy_(list_g[0].data)
        else:
            cur = 0
            for i in range(self.num_var):
                self.gradient_tensor[cur:(cur + self.store_lens[i])] = list_g[i].data.view(self.store_lens[i])
                cur = cur + self.store_lens[i]
        return()
    def load_Hessian(self,H):
        if not self.need_flatten:
            self.Hessian_tensor.copy_(H.data)
        else:
            for i in range(self.num_var):
                for j in range(self.num_var):
                    # where to put the flattened tensor in the Hessian
                    self.Hessian_tensor[self.store_slices[i],self.store_slices[j]]= \
                        self.flatten_to_tensor(H[i,j],shape=(self.store_lens[i],self.store_lens[j]))
        return()
    def load_dH(self,dH):
        if not self.need_flatten:
            self.dH_tensor.copy_(dH)
        else:
            for i in range(self.num_var):
                for j in range(self.num_var):
                    for k in range(self.num_var):
                        # where to put the flattened tensor in the dH tensor
                        #out = self.flatten_to_tensor(dH[i,j,k],shape=(self.store_lens[i],self.store_lens[j],self.store_lens[k]))
                        #print(out)
                        self.dH_tensor[self.store_slices[i],self.store_slices[j],self.store_slices[k]] = \
                            self.flatten_to_tensor(dH[i,j,k],shape=(self.store_lens[i],self.store_lens[j],self.store_lens[k]))
        return()

    def load_mdiagH(self,mdiagH):
        cur = 0
        for i in range(self.num_var):
            self.mdiagH_tensor[cur:(cur + self.store_lens[i])] = mdiagH[i].data.view(self.store_lens[i])
            cur = cur + self.store_lens[i]
        return()
    def load_mgraddiagH(self,mgraddiagH):
        for i in range(self.num_var):
            for j in range(self.num_var):
                # where to put the flattened tensor in the Hessian
                self.mgraddiagH_tensor[self.store_slices[i],self.store_slices[j]]= \
                    self.flatten_to_tensor(mgraddiagH[i,j],shape=(self.store_lens[i],self.store_lens[j]))
        return()
    def V_higherorder_setup(self):
        self.gradient_tensor = torch.zeros(self.dim)
        self.list_var = list(self.parameters())
        if self.need_higherorderderiv == True:
            self.diagH_tensor = torch.zeros(self.dim)
            self.graddiagH_tensor = torch.zeros(self.dim, self.dim)
            self.Hessian_tensor = torch.zeros((self.dim, self.dim))
            self.dH_tensor = torch.zeros((self.dim, self.dim, self.dim))
            #if self.metric.name == "softabs_diag":
            #    self.mdiagH_tensor = torch.zeros(self.dim)
            #    self.mgraddiagH_tensor = torch.zeros(self.dim, self.dim)
            #else:
            #    self.Hessian_tensor = torch.zeros((self.dim,self.dim))
            #   self.dH_tensor = torch.zeros((self.dim,self.dim,self.dim))


        return()
    def flatten_to_tensor(self,obj,shape):
        # take entry in the abstract block Hessian or abstract block dH and return a block in the
        # flattened Hessian or dH
        store = torch.zeros(shape)

        if len(shape)==2:
            obj = numpy.reshape(obj, [shape[0]])
            for i in range(shape[0]):
                #print(obj[i])
                #exit()
                store[i,:]= obj[i].data.view(-1)
        elif len(shape)==3:
      #      print("yes")
            obj = numpy.reshape(obj,[shape[0]])
            for i in range(shape[0]):
                obj[i] = numpy.reshape(obj[i],[shape[1]])
            for i in range(shape[0]):
                for j in range(shape[1]):
                    store[i,j,:] = obj[i][j].data.view(-1)
        else:
            raise ValueError("")
        return (store)

    def load_flattened_tensor_to_param(self,flattened_tensor=None):
        cur = 0
        if flattened_tensor is None:
            for i in range(self.num_var):
                # convert to copy_ later
                self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i])
        else:
            for i in range(self.num_var):
                # convert to copy_ later
                self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i])
        return()
    def decides_if_flattened(self):
        self.need_flatten = False
        self.list_var = list(self.parameters())
        self.num_var = len(self.list_var)
        self.list_tensor = numpy.empty(self.num_var,dtype=type(self.list_var[0].data))
        for i in range(len(self.list_tensor)):
            self.list_tensor[i] = self.list_var[i].data
        if self.num_var>1:
            self.need_flatten = True
        elif self.num_var==1:
            if len(list(self.list_var[0].data.shape))>1:
                self.need_flatten = True
        else:
            raise ValueError("count is >=2 but not 1")
        self.store_shapes,self.store_lens,self.dim,self.store_slices=get_list_stats(self.list_var)
        if self.need_flatten:
            self.flattened_tensor = torch.zeros(self.dim)
            cur = 0
            for i in range(self.num_var):
                self.flattened_tensor[cur:(cur + self.store_lens[i])] = self.list_var[i].data.view(self.store_lens[i])
                cur = cur + self.store_lens[i]
        else:
            self.flattened_tensor = self.list_var[0].data
        return()

    def dq(self,p_flattened_tensor):
        self.load_flattened_tensor_to_param(p_flattened_tensor)
        g = grad(self.forward(), self.list_var)
        out = torch.zeros(len(p_flattened_tensor))
        cur = 0
        for i in range(self.num_var):
            out[cur:(cur + self.store_lens[i])] = g[i].data.view(self.store_lens[i])
            cur = cur + self.store_lens[i]
        return(out)
    def create_T(self,metric):
        # metric object- includes information about type and cov or diag_v, or softabs
        self.T = T(metric)
        # might be problematic -- recrusive container
        self.T.linkedV = self

        return()
    def load_point(self,q_point):
        for i in range(self.num_var):
            # convert to copy_ later
            self.list_var[i].data.copy_(q_point.list_var[i].data)
        self.flattened_tensor.copy_(q_point.flattened_tensor)

        return()









