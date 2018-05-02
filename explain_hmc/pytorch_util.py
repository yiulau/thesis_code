import torch,numpy


def get_list_stats(list_var):
    num_var = len(list_var)
    store_lens = [None] * num_var
    store_shape = [None] * num_var
    store_slices = [0] * num_var
    dim = 0
    cur = 0
    for i in range(num_var):
        store_shape[i] = list(list_var[i].data.shape)
        store_lens[i] = int(numpy.prod(store_shape[i]))
        dim += store_lens[i]
        store_slices[i] = numpy.s_[cur:(cur+store_lens[i])]
        cur = cur+store_lens[i]
    return (store_shape, store_lens, dim,store_slices)