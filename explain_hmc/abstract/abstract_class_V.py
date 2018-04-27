import torch,abc
from torch.autograd import Variable, grad
import torch.nn as nn
import numpy
from pytorch_util import get_list_stats
from explicit.genleapfrog_ult_util import eigen

# if need to define explicit gradient do it

class V(nn.Module):
    #def __init__(self,explicit_gradient):
    def __init__(self):
        super(V, self).__init__()
        print("yes")
        self.V_setup()
        if self.explicit_gradient==None:
            raise ValueError("self.explicit_gradient need to be defined in V_setup")
        if self.need_higherorderderiv==None:
            raise ValueError("self.need_higherorderderiv need to be defined in V_setup")

        ##################################################################################
        self.decides_if_flattened()
        self.V_higherorder_setup()
        self.point_generator = point(V=self)
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
    def forward_float(self):
        return(self.forward().data[0])
    def gradient(self):
        # load gradients to list(self.parameters()).grad
        if not self.explicit_gradient:
            o = self.forward()
            o.backward()
        else:
            self.load_explicit_gradient()
    def getdV(self):
        #
        # return list of pytorch variables containing the gradient
        if not self.need_flatten:
            g = grad(self.forward(),self.list_var,create_graph=True)[0]
        else:
            g = grad(self.forward(), self.list_var, create_graph=True)
        #self.load_gradient(g)
        return(g)

    def getdV_tensor(self,q):
        # return list of pytorch variables containing the gradient
        if not q.V is self:
            raise ValueError("not the same V function object")
        else:
            q.load
        if not self.need_flatten:
            g = grad(self.forward(),self.list_var,create_graph=True)[0]
        else:
            g = grad(self.forward(), self.list_var, create_graph=True)
        self.load_gradient(g)
        return(self.gradient_tensor)

    def getH(self,q):
        if not self.need_flatten:
            g = self.getdV()
            H = Variable(torch.zeros(self.dim, self.dim))
            for i in range(self.dim):
                self.H[i, :] = grad(g[i], self.list_var, create_graph=True)[0]
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
    def getH_tensor(self,q):
        if not self.need_flatten:
            g = self.getdV()
            H = Variable(torch.zeros(self.dim, self.dim))
            for i in range(self.dim):
                self.H[i, :] = grad(g[i], self.list_var, create_graph=True)[0]
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

    def getdH(self):
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
    def getdH_tensor(self,q):
        # takes anything but outputs tensor
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
        self.load_gradient(g)
        self.load_Hessian(H)
        self.load_dH(dH)

        return(self.gradient_tensor,self.Hessian_tensor,self.dH_tensor)

    def get_mdiagH(self,q):
        g,H = self.getH()
        mdiagH = numpy.diag(H)
        return(g,mdiagH)
    def getdH_diagonal_tensor(self,q):
        #returns (dV,mdiagH,mgraddiagH)
        self.load_point(q)
        if not self.need_flatten:
            g, mdiagH = self.getmdiagH(q)
            mgraddiagH = torch.zeros(self.dim, self.dim, self.dim)
            for i in range(self.dim):
                    mgraddiagH[:, i] = grad(mdiagH[i], self.list_var, create_graph=False, retain_graph=True)[0].data
        else:
            g, mdiagH = self.getmdiagH()
            mgraddiagH = numpy.empty(self.num_var, self.num_var)
            for i in range(self.num_var):
                for j in range(self.num_var):
                    mgraddiagH[i,j] = self.block_2nd_deriv(mgraddiagH[i],self.list_var[j],True,True)

        self.load_graident(g)
        self.load_mdiagH(mdiagH)
        self.load_mgraddiagH(mgraddiagH)

        return(self.gradient_tensor,self.mdiagH_tensor,self.mgraddiagH_tensor)
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
        cur = 0
        for i in range(self.num_var):
            self.gradient_tensor[cur:(cur + self.store_lens[i])] = list_g[i].data.view(self.store_lens[i])
            cur = cur + self.store_lens[i]
        return()
    def load_Hessian(self,H):
        for i in range(self.num_var):
            for j in range(self.num_var):
                # where to put the flattened tensor in the Hessian
                self.Hessian_tensor[self.store_slices[i],self.store_slices[j]]= \
                    self.flatten_to_tensor(H[i,j],shape=(self.store_lens[i],self.store_lens[j]))
        return()
    def load_dH(self,dH):
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
        if self.metric.name=="softabs_diag":
            self.mdiagH_tensor = torch.zeros(self.dim)
            self.mgraddiagH_tensor = torch.zeros(self.dim,self.dim)
        if self.need_higherorderderiv == True:
            self.Hessian_tensor = torch.zeros((self.dim,self.dim))
            self.dH_tensor = torch.zeros((self.dim,self.dim,self.dim))
            self.list_var = list(self.parameters())

        return()
    def flatten_to_tensor(self,obj,shape):
        store = torch.zeros(shape)

        if len(shape)==2:
            obj = numpy.reshape(obj, [shape[0]])
            for i in range(shape[0]):
                #print(obj[i])
                #exit()
                store[i,:]= obj[i].data.view(-1)
        elif len(shape)==3:
            print("yes")
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
        if flattened_tensor==None:
            for i in range(self.num_var):
                # convert to copy_ later
                self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shape[i])
        else:
            for i in range(self.num_var):
                # convert to copy_ later
                self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shape[i])
        return()
    def decides_if_flattened(self):
        self.need_flatten = False
        list_var = list(self.parameters())
        self.num_var = len(list_var)
        if self.num_var>1:
            self.need_flatten = True
        elif self.num_var==1:
            if len(list(list_var[0].data.shape))>1:
                self.need_flatten = True
        else:
            raise ValueError("count is >=2 but not 1")
        self.store_shape,self.store_lens,self.dim,self.store_slices=get_list_stats(list_var)
        if self.need_flatten:
            self.flattened_tensor = torch.zeros(self.dim)
            cur = 0
            for i in range(self.num_var):
                self.flattened_tensor[cur:(cur + self.store_lens[i])] = list_var[i].data.view(self.store_lens[i])
                cur = cur + self.store_lens[i]
        else:
            self.flattened_tensor = list_var[0].data
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

class T(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,metric):
        self.metric = metric
        self.dim = self.linkedV.dim
        self.list_lens = self.linkedV.list_lens
        self.list_shapes = self.linkedV.list_shapes

        self.list_tensor = numpy.empty(len(self.list_shapes),dtype=self.linkedV.list_tensor[0].type())
        for i in range(len(self.list_tensor)):
            self.list_tensor[i] = torch.zeros(self.linkedV.list_tensor[i].shape)
        self.p = point(self)
        # need gradient for every metric except unit_e
        self.gradient = numpy.empty(len(self.list_shapes), dtype=torch.FloatTensor)
        # definitely need faltten for softabs,
        # for other metrics depend on
        if self.need_flatten:
            self.flattened_gradient = torch.zeros(self.dim)
        else:
            if self.metric.name=="dense_e":
                self.flattened_gradient = self.p

    def load_listp_to_flattened(self,list_tensor,target_tensor):
        cur = 0
        for i in range(len(self.p)):
            target_tensor[cur:(cur + self.list_lens[i])] = self.list_tensor[i].view(self.list_lens[i])
            cur = cur + self.list_lens[i]
    def load_flattened_tenosr_to_target_list(self,target_list,flattened_tensor):
        cur = 0
        for i in range(len(target_list)):
            target_list[i].copy_(flattened_tensor[cur:(cur + self.list_lens[i])].view(self.list_shapes[i]))
            cur = cur + self.list_lens[i]
        return ()

    #@abc.abstractmethod
    #def evaluate_float(self):
        # returns T(q,p) -- float
    #    return()

    #@abc.abstractmethod
    #def dp(self):
        # takes and returns flattened_tensor
    #    return()
    #@abc.abstractmethod
    #def dtaudp(self,lam=None,Q=None):
    #    return()

    #@abc.abstractmethod
    #def dtaudq(self,alpha=None,H_=None,dH=None):
        # flattened_p,q
     #   return()

    #@abc.abstractmethod
    #def generate_momentum(self,lam=None,Q=None):
        # returns generated in a list that has the same shape as the original variables
     #   return(self.store_momentum)


class H(object):
    # hamiltonian function
    def __init__(self,V,T):
        self.V = V
        self.T = T
        self.dG_dt = self.setup_dG_dt()
        self.p_sharp = self.setup_p_sharp()
    def evaluate(self):
        out = V.evaluate() + T.evaluate_float()
        return(out)

    def setup_dG_dt(self):
        if (self.metric.name == "softabs"):
            def dG_dt(q,p,dV=None,lam=None,Q=None,dH=None):
                if dV==None:
                    dV, H_, dH = self.V.getdH_tensor(q)
                    lam,Q = eigen(H_)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p.flattened_tensor, dH, Q, lam, alpha)) + self.V.dphidq(lam, alpha, dH, Q, dV))
        elif (self.metric.name == "softabs_diag"):
            def dG_dt(q, p,dV=None,mdiagH=None,mgraddiagH=None):
                if dV==None:
                    dV, mdiagH, mgraddiagH = self.V.getdH_diagonal_tensor(q)
                mlambda,_ = self.T.fcomputeMetric(mdiagH)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p.flattened_tensor,mdiagH,mlambda,mgraddiagH)) +
                        self.V.dphidq(mdiagH,mlambda))
        elif (self.metric.name=="softabs_diag_outer_product" ):
            def dG_dt(q, p,dV=None):
                if dV==None:
                    dV = self.V.getdV_tensor(q)
                mlambda, _ = self.T.fcomputeMetric(dV)
                mH = self.mH(dV)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p, dV, mlambda,mH)) +
                        self.V.dphidq(p.flattened_tensor,mlambda,dV,mH))
        elif (self.metric.name=="softabs_outer_product" ):
            def dG_dt(q, p,dV=None):
                if dV==None:
                    dV = self.V.getdV_tensor(q)
                    mH = self.mH(dV)
                alpha = self.T.metric.msoftabsalpha
                return (-torch.dot(q.flattened_tensor, self.V.dtaudq(p.flattened_tensor,dV)) +
                        self.V.dphidq(dV,mH))
        else:
            def dG_dt(q,p):
                return (2 * self.T.evaluate(p) - torch.dot(q.flattened_tensor, self.V.dq(q)))

    def set_p_sharp(self):
        if (self.metric == "softabs"):
            def p_sharp(q,p,lam=None,Q=None):
                out = p.point_clone()
                out.flattened_tensor = T.dtaudp(p.flattened_tensor,lam,Q)
                out.loadfromflatten()
                return(out)
        elif(self.metric.name=="softabs_diag"):
            def p_sharp(q,p,mlambda=None):
                out = p.point_clone()
                out.flattened_tensor = T.dtaudp(p.flattened_tensor,mlambda)
                out.loadfromflatten()
                return(out)
        elif(self.metric.name=="softabs_outer_product" or self.metric.name=="softabs_diag_outer_product"):
            def p_sharp(q,p,dV=None):
                out = p.point_clone()
                out.flattened_tensor = T.dtaudp(p.flattened_tensor,dV)
                out.loadfromflatten()
                return(out)

        else:
            def p_sharp(q,p):
                out = p.point_clone()
                out.flattened_tensor = T.dp(p.flattened_tensor)
                out.loadfromflatten()
                return(out)

        return(p_sharp)


class point(object):
    # numpy vector of pytorch variables
    # need to be able to calculate  norm(q-p)_2 (esjd)
    # sum two points
    # clone a point
    # multiply by scalar
    # sum load (target, summant)
    # return list of tensors
    # dot product

    def __init__(self,V=None,T=None,need_var=None):
        self.V = V
        self.T = T
        if self.V==None:
            self.need_var = True
            self.list_var = self.T.list_var
            self.list_tensor = self.T.list_tensor
            self.flattened_tensor = self.T.flattened_tensor

        elif self.T==None:
            self.need_var = True
            self.list_var = self.V.list_var
            self.list_tensor = self.V.list_tensor
            self.flattened_tensor = self.V.flattened_tensor
        else:
            raise ValueError("deal with this later")
    def dot_product(self,another_point):
        out = 0
        for i in range(self.num_var):
            out += (self.list_var[i].data * self.another_point[i].data).sum()
        return(out)
    def diff_square(self,another_point):
        # returns sum((self-another_point)^2)
        out = 0
        for i in range(self.num_var):
            temp = self.list_var[i].data-another_point.list_var[i].data
            out += (temp*temp).sum()
        return(out)
    def varclone(self):
        out = numpy.empty_like(self.list_var)
        for i in range(self.num_var):
            out[i] = Variable((self.list_var[i].data),self.list_var[i].requires_grad)
        return(out)
    def tensorclone(self):
        out = numpy.empty_like(self.list_tensor)
        for i in range(self.num_var):
            out[i] = self.list_tensor[i].clone()
        return(out)
    def zeroclone(self):
        # creates list with same shape but filled with zeros
        out = numpy.empty_like(self.list_var)
        for i in range(self.num_var):
            out[i] = Variable(torch.zeros(self.list_shapes[i]),requires_grad=self.list_var[i].requires_grad)
        return(out)
    def sum_load(self,another_point):
        for i in range(self.num_var):
            self.list_var[i].data += another_point[i].data
        return()

    def multiply_by_scalar(self,scalx):
        if self.dtype == torch.tensor:
            out = self.tensor_clone()
            for i in range(self.num_var):
                out[i] *= scalx
        else:
            out = self.npclone()
            for i in range(self.num_var):
                out[i].data *= scalx
        return(out)
    def load_flatten(self):
        if self.need_flatten:
            self.load_flattened_tensor_to_param()
        else:
            pass
        return()
    def sum_point(self,another_point):
        # returns the sum of two points
        out = self.npclone()
        self.sum_load(out,another_point)
        return(out)

    def load_flattened_tensor_to_param(self,flattened_tensor=None):
        cur = 0
        if flattened_tensor==None:
            for i in range(self.num_var):
                # convert to copy_ later
                self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shape[i])
        else:
            for i in range(self.num_var):
                # convert to copy_ later
                self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shape[i])
        return()






