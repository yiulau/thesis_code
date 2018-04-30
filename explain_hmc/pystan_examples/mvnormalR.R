# Compare RW-Metropolis with HMC
library(mvtnorm)
cov_matrix = matrix(c(1,0.9,0.9,1),ncol=2,nrow=2)
chain_L=500
library('rstan')
rstan_options(auto_write = TRUE)

mod = stan_model(file="mvnormal.stan")

data = list(N=dim(cov_matrix)[1],mu=rep(0,dim(cov_matrix)[1]),Sigma=cov_matrix)
fit = sampling(mod,data=data,iter=chain_L,chain=1,algorithm="HMC",control=list(metric="unit_e",stepsize_jitter=0,stepsize=0.1,int_time=10,adapt_engaged=F))
out = extract(fit,pars="y")
plot(x=1:length(out[[1]][,1]),y=out[[1]][,1])

metropolis = function(sigma2,chain_L,initx,mu,Sigma){
  accept_v = rep(0,chain_L)
  storematrix=matrix(0,ncol=length(initx),nrow=chain_L)
  storematrix[1,]=initx
  for(i in 2:chain_L){
    prox = storematrix[i-1,] + rnorm(n=length(initx))*sqrt(sigma2)
    d_prox = dmvnorm(x=prox,mean=mu,sigma=Sigma,log=T)
    d_curx = dmvnorm(x=storematrix[i-1,],mean=mu,sigma=Sigma,log=T)
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
rw_stepsize = 1.8
p = dim(cov_matrix)[1]
out_rw = metropolis(sigma2 = rw_stepsize,chain_L = chain_L,initx=rep(0,p),mu=rep(0,p),Sigma=cov_matrix)
out_rw[[1]]
plot(out_rw[[2]][,2])