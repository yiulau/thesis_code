#sample from banana distribution
library('rstan')
rstan_options(auto_write = TRUE)
chain_L=1500
mod = stan_model(file="banana.stan")
fit = sampling(mod,iter=chain_L,chain=1,algorithm="HMC",control=list(metric="unit_e",stepsize_jitter=0,stepsize=0.1,int_time=10,adapt_engaged=F))

out = extract(fit,pars="y")
plot(out[[1]][,1])
plot(out[[1]])
# RW
log_d_banana=function(x){
  -(100*(x[2]-x[1^2]^2)^2+(1-x[1])^2)/20
}
metropolis_banana = function(sigma2,chain_L,initx){
  accept_v = rep(0,chain_L)
  storematrix=matrix(0,ncol=length(initx),nrow=chain_L)
  storematrix[1,]=initx
  for(i in 2:chain_L){
    prox = storematrix[i-1,] + rnorm(n=length(initx))*sqrt(sigma2)
    d_prox = log_d_banana(prox)
    d_curx = dmvnorm(storematrix[i-1,])
    accept = runif(1) < exp(d_prox-d_curx)
    if(accept){
      storematrix[i,] = prox
    }else{
      storematrix[i,] = storematrix[i-1,]
    }
    accept_v[i]=accept
  }
  accept_rate = sum(accept_v)/chain_L
  return(list(accept_rate,storematrix))
}
rw_stepsize = 0.3
p = dim(cov_matrix)[1]
out_rw = metropolis_banana(sigma2 = rw_stepsize,chain_L = chain_L,initx=rep(0,2))
out_rw[[1]]
plot(out_rw[[2]][,2])