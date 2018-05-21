from torch.autograd import Variable
import numpy, torch
# how a point is defined depends on the problem at hand. if nn then we might not want an extra copy of the weights
class point(object):
    # numpy vector of pytorch variables
    # need to be able to calculate  norm(q-p)_2 (esjd)
    # sum two points
    # clone a point
    # multiply by scalar
    # sum load (target, summant)
    # return list of tensors
    # dot product
    # need_flatten depends on metric. anything other than unit_e and diag_e need_flatten = True

    def __init__(self,V=None,T=None,list_var=None,list_tensor=None,pointtype=None,
                 need_flatten=True):
        self.V = V
        self.T = T
        self.list_var= list_var
        self.list_tensor = list_tensor
        self.pointtype = pointtype
        self.need_flatten = need_flatten
        if not V is None:
            self.pointtype = "q"
            target_fun_obj = V
            self.need_flatten = target_fun_obj.need_flatten
            self.num_var = target_fun_obj.num_var
            source_list_tensor = target_fun_obj.list_tensor
        elif not T is None:
            self.pointtype = "p"
            target_fun_obj = T
            self.need_flatten = target_fun_obj.need_flatten
            self.num_var = target_fun_obj.num_var
            source_list_tensor = target_fun_obj.list_tensor
        else:
            if list_var is None and list_tensor is None:
                raise ValueError("one of V,T or a list of variables, or a list of tensors must be supplied")
            else:
                if not list_var is None and not list_tensor is None:
                    raise ValueError("can't supply both list_tensor and list_var. need exactly one ")
                assert pointtype in ("p","q")
                self.pointtype = pointtype

                assert not list_tensor is None
                self.num_var = len(list_tensor)
                if list_tensor is None:
                    source_list_tensor = []
                    assert not list_var is None
                    for i in range(len(list_var)):
                        source_list_tensor.append(list_var[i].data)
                else:
                    source_list_tensor = list_tensor





        self.list_tensor = numpy.empty(len(source_list_tensor),dtype=type(source_list_tensor[0]))
        for i in range(len(source_list_tensor)):
            self.list_tensor[i] = source_list_tensor[i].clone()
        self.store_lens = []
        self.store_shapes = []
        for i in range(len(self.list_tensor)):
            shape = list(self.list_tensor[i].shape)
            length = int(numpy.prod(shape))
            self.store_shapes.append(shape)
            self.store_lens.append(length)
        self.dim = sum(self.store_lens)
        if need_flatten:
            self.flattened_tensor = torch.zeros(self.dim)
            self.load_param_to_flatten()
        else:
            assert len(self.list_tensor[0])==self.dim
            self.flattened_tensor = self.list_tensor[0]
        self.syncro = self.assert_syncro()
    def load_flatten(self):
        if self.need_flatten:
            self.load_flattened_tensor_to_param()
        else:
            pass
        return()

    def point_clone(self):
        out = point(V=self.V,T=self.T,list_var=self.list_var,list_tensor=self.list_tensor,pointtype=self.pointtype)
        return(out)

    def assert_syncro(self):
        # check that list_tensor and flattened_tensor hold the same value
        temp = self.flattened_tensor.clone()
        cur = 0
        for i in range(self.num_var):
            temp[cur:(cur + self.store_lens[i])].copy_(self.list_tensor[i].view(self.store_shapes[i]))
            cur = cur + self.store_lens[i]
        diff = ((temp - self.flattened_tensor) * (temp - self.flattened_tensor)).sum()
        if diff > 1e-6:
            out = False
        else:
            out = True
        return (out)

    def load_flattened_tensor_to_param(self):
        cur = 0
        for i in range(self.num_var):
            self.list_tensor[i].copy_(self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i]))
            cur = cur + self.store_lens[i]

    def load_param_to_flatten(self):
        cur = 0
        for i in range(self.num_var):
            self.flattened_tensor[cur:(cur + self.store_lens[i])].copy_(self.list_tensor[i].view(-1))
            cur = cur + self.store_lens[i]


    #def dot_product(self,another_point):
    #    out = 0
    #    for i in range(self.num_var):
    #        out += (self.list_var[i].data * self.another_point[i].data).sum()
    #    return(out)
    #def diff_square(self,another_point):
    #    # returns sum((self-another_point)^2)
    #    out = 0
    #    for i in range(self.num_var):
    #        temp = self.list_var[i].data-another_point.list_var[i].data
    #        out += (temp*temp).sum()
    #    return(out)
    #def varclone(self):
    #    out = numpy.empty_like(self.list_var)
    #    for i in range(self.num_var):
    #        out[i] = Variable((self.list_var[i].data),self.list_var[i].requires_grad)
    #    return(out)
    #def tensorclone(self):
    #    # creates list with same shape and filled with clones of tensor
    #    out = numpy.empty_like(self.list_tensor)
    #    for i in range(self.num_var):
    #        out[i] = self.list_tensor[i].clone()
    #    return(out)
    #def zeroclone(self):
    #    # creates list with same shape but filled with zeros
    #    out = numpy.empty_like(self.list_var)
    #    for i in range(self.num_var):
    #        out[i] = Variable(torch.zeros(self.store_shapes[i]),requires_grad=self.list_var[i].requires_grad)
    #    return(out)
    #def sum_load(self,another_point):
    #    for i in range(self.num_var):
    #        self.list_var[i].data += another_point[i].data
    #    return()

    #def multiply_by_scalar(self,scalx):
    #    if self.dtype == torch.tensor:
     #       out = self.tensor_clone()
     #       for i in range(self.num_var):
     #           out[i] *= scalx
     #   else:
     #       out = self.npclone()
     #       for i in range(self.num_var):
     #           out[i].data *= scalx
     #   return(out)

    #def sum_point(self,another_point):
    #    # returns the sum of two points
    #    out = self.npclone()
    #    self.sum_load(out,another_point)
    #    return(out)

    #def load_flattened_tensor_to_param(self,flattened_tensor=None):
    #    cur = 0
    #    if flattened_tensor is None:
    #        for i in range(self.num_var):
     #           # convert to copy_ later
     #           self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i])
     #   else:
     #       for i in range(self.num_var):
     #           # convert to copy_ later
     #           self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i])
     #   return()






class point_deprecated(object):
    # numpy vector of pytorch variables
    # need to be able to calculate  norm(q-p)_2 (esjd)
    # sum two points
    # clone a point
    # multiply by scalar
    # sum load (target, summant)
    # return list of tensors
    # dot product

    def __init__(self,V=None,T=None,list_var=None,list_tensor=None,flattened_tensor=None,need_var=None):
        self.V = V
        self.T = T
        if not self.V==None:
            self.pointtype = "q"
        elif not self.T is None:
            self.pointtype = "p"
        else:
            raise ValueError("one of V,T must be supplied")
        if self.pointtype=="p":
            self.need_flatten = self.T.need_flatten
            self.num_var = self.T.num_var
            if list_var is None:
                source_list_var = self.T.list_var
            else:
                source_list_var = list_var
            if list_tensor is None:
                source_list_tensor = self.T.list_tensor
            else:
                source_list_tensor = list_tensor
            self.need_var = True
            self.list_var = numpy.empty(len(source_list_var), dtype=type(source_list_var[0]))
            for i in range(len(source_list_var)):
                self.list_var[i] = Variable(source_list_var[i].data.clone(), requires_grad=False)
            #self.list_var = self.T.list_var
            self.list_tensor = numpy.empty(len(source_list_tensor),dtype=type(source_list_tensor[0]))
            for i in range(len(self.T.list_tensor)):
                self.list_tensor[i] = source_list_tensor[i].clone()
            if flattened_tensor is None:
                self.flattened_tensor = self.T.flattened_tensor.clone()
            else:
                self.flattened_tensor = flattened_tensor.clone()

        elif self.pointtype=="q":
            self.need_flatten = self.V.need_flatten
            self.num_var = self.V.num_var
            self.need_var = True
            if list_var is None:
                source_list_var = self.V.list_var
            else:
                source_list_var = list_var
            if list_tensor is None:
                source_list_tensor = self.V.list_tensor
            else:
                source_list_tensor = list_tensor
            self.list_var = numpy.empty(len(source_list_var), dtype=type(source_list_var[0]))
            for i in range(len(source_list_var)):
                self.list_var[i] = Variable(source_list_var[i].data.clone(), requires_grad=False)
            # self.list_var = self.T.list_var
            self.list_tensor = numpy.empty(len(source_list_tensor), dtype=type(source_list_tensor[0]))
            for i in range(len(self.list_tensor)):
                self.list_tensor[i] = source_list_tensor[i].clone()
            if flattened_tensor is None:
                self.flattened_tensor = self.V.flattened_tensor.clone()
            else:
                self.flattened_tensor = flattened_tensor.clone()
        else:
            raise ValueError("deal with this later")
    #def dot_product(self,another_point):
    #    out = 0
    #    for i in range(self.num_var):
    #        out += (self.list_var[i].data * self.another_point[i].data).sum()
    #    return(out)
    #def diff_square(self,another_point):
    #    # returns sum((self-another_point)^2)
    #    out = 0
    #    for i in range(self.num_var):
    #        temp = self.list_var[i].data-another_point.list_var[i].data
    #        out += (temp*temp).sum()
    #    return(out)
    #def varclone(self):
    #    out = numpy.empty_like(self.list_var)
    #    for i in range(self.num_var):
    #        out[i] = Variable((self.list_var[i].data),self.list_var[i].requires_grad)
    #    return(out)
    #def tensorclone(self):
    #    # creates list with same shape and filled with clones of tensor
    #    out = numpy.empty_like(self.list_tensor)
    #    for i in range(self.num_var):
    #        out[i] = self.list_tensor[i].clone()
    #    return(out)
    #def zeroclone(self):
    #    # creates list with same shape but filled with zeros
    #    out = numpy.empty_like(self.list_var)
    #    for i in range(self.num_var):
    #        out[i] = Variable(torch.zeros(self.store_shapes[i]),requires_grad=self.list_var[i].requires_grad)
    #    return(out)
    #def sum_load(self,another_point):
    #    for i in range(self.num_var):
    #        self.list_var[i].data += another_point[i].data
    #    return()

    #def multiply_by_scalar(self,scalx):
    #    if self.dtype == torch.tensor:
     #       out = self.tensor_clone()
     #       for i in range(self.num_var):
     #           out[i] *= scalx
     #   else:
     #       out = self.npclone()
     #       for i in range(self.num_var):
     #           out[i].data *= scalx
     #   return(out)
    def load_flatten(self):
        if self.need_flatten:
            self.load_flattened_tensor_to_param()
        else:
            pass
        return()
    #def sum_point(self,another_point):
    #    # returns the sum of two points
    #    out = self.npclone()
    #    self.sum_load(out,another_point)
    #    return(out)

    #def load_flattened_tensor_to_param(self,flattened_tensor=None):
    #    cur = 0
    #    if flattened_tensor is None:
    #        for i in range(self.num_var):
     #           # convert to copy_ later
     #           self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i])
     #   else:
     #       for i in range(self.num_var):
     #           # convert to copy_ later
     #           self.list_var[i].data = self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i])
     #   return()

    def point_clone(self):
        out = point(V=self.V,T=self.T,list_var=self.list_var,list_tensor=self.list_tensor,flattened_tensor=self.flattened_tensor,need_var=self.need_var)
        return(out)
