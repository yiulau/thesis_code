library(RcppCNPy)
library(coda)
library(mcmcse)


fmat = array(0,c(2,40,4))
for(i in 1:2){
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
plot(list_obj)

# gelman statistics
gelman.diag(list_obj)
