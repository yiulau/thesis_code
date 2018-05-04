library(RcppCNPy)
library(coda)
library(mcmcse)

shape = npyLoad("shape.npy")

fmat = array(0,shape)
for(i in 1:shape[1]){
  name = paste(c("temp",i-1,".npy"),collapse="")
  fmat[i,,] = npyLoad(name)
}
#fmat <- npyLoad("temp.npy")
#print(fmat[1,])
#dim(fmat)
num_chains = dim(fmat)[1]
store_list = vector("list",num_chains)
for(i in 1:num_chains){
  store_list[[i]] = mcmc(fmat[i,,])
}

flattened_mat = matrix(fmat,ncol=dim(fmat)[3])

list_obj = mcmc.list(store_list)
# effective sample size 
ess(flattened_mat)
effectiveSize(list_obj)
# traceplots 
#plot(list_obj)

# gelman statistics
out = gelman.diag(list_obj)
gelman_matrix = as.matrix(out[[1]])

# write to numpy file
npySave("routput.npy",gelman_matrix)
