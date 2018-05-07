import numpy,torch,math
class metric(object):
    # should store information like Cov, vec. Whether its for flattened tensor defined V. The cov and var should be in
    # flattened version if necessary
    # stores alpha for softabs
    def __init__(self,name,V_instance,alpha=None):

        self.name = name
        self.criterion = criterion
        if not self.criterion=="gnuts" or not self.criterion=="nuts" or not self.criterion=="xhmc":
            raise ValueError("unknown termination criterion")
        if self.name=="unit_e":
            pass
        elif self.name=="diag_e":
            self.num_var = V_instance.num_var
            self.store_shapes = V_instance.store_shapes
            self.store_lens = V_instance.store_lens
            self._sd_list_tensor = numpy.empty(self.num_var, dtype=type(V_instance.flattened_tensor))
            for i in range(self.num_var):
                self._sd_list_tensor[i] = torch.ones(self.store_shapes[i])
            self._var_list_tensor = numpy.empty(self.num_var,dtype=type(V_instance.list_var[0]))
            for i in range(self.num_var):
                self._var_list_tensor[i] = torch.ones(self.store_shapes[i])
            self._var_vec = torch.ones(V_instance.dim)
            self._sd_vec = torch.ones(V_instance.dim)
        elif name=="dense_e":
            # covL * covL^T = cov
            self._flattened_covL = torch.eye(V_instance.dim,V_instance.dim)
            self._flattened_cov = torch.eye(V_instance.dim,V_instance.dim)
        elif name=="softabs" or name=="softabs_diag" or name=="softabs_outer_product" or name=="softabs_outer_product_diag":
            if alpha==None:
                raise ValueError("alpha needs be defined for softabs metric")
            elif alpha <= 0 or alpha==math.inf:
                raise ValueError("alpha needs be > 0 and less than < Inf")
            self.msoftabsalpha = alpha

        else:
            raise ValueError("unknown metric type")
    def set_metric(self,var_or_cov_tensor):
        # input: either flattened empircial covariance for dense_e or
        # list of variances with the shape of p for diag_e
        if self.name == "diag_e":
            for i in range(self.num_var):
                self._sd_list[i].copy_(torch.sqrt(var_or_cov_tensor[i]))
                self._var_list[i].copy_(var_or_cov_tensor[i])
        elif self.name == "dense_e":
            raise ValueError("fix this")
            self._flattened_covL.copy_(sd_or_covL_tensor)
        else:
            raise ValueError("should not use this function unless the metrics are diag_e or dense_e")

    def initialize_m_m_2(self,point):
        m_ = point.zeroclone()
        if self.name=="dense_e":
            m_2 = torch.zeros((self.dim,self.dim))
        elif self.name=="diag_e":
            m_2 = point.zeroclone()
        return(m_,m_2)

