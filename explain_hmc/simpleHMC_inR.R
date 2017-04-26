HMC = function(U,gradU,epsilon,L,currentq){
  q = currentq
  p = rnorm(length(q),0,1)
  currentp = p
  
  p = p - epsilon * gradU(q)/2
  i = 1
  for (i in 1:L){
    q = q + epsilon * p
    if (i!=L) p = p - epsilon * gradU(q)
  }
  
  p = p - epsilon * gradU(q) / 2
  p = -p 
  
  curU = U(currentq)
  curK = sum(currentp^2)/2
  proU = U(q)
  proK = sum(p^2)/2
  
  if(runif(1)<exp(curU-proU+curK-proK)){
    return(q)
  }else{
    return(currentq)
  }
}
# negative log density
target_f = function(x){
  return(-dnorm(x,log=T))
}
# gradient of negative log
target_grad = function(x){
  return(x)
}
# Start sampling
num_draws = 100
epsilon = 1
L = 10
initq = 1
storematrix = array(0,dim=c(num_draws,2))
storematrix[1,2]=initq

for(i in 2:num_draws){
  out = HMC(U=target_f,gradU=target_grad,epsilon=epsilon,L=L,currentq=storematrix[i-1,2])
  if(out!=storematrix[i-1,2]){
    storematrix[i,1]=1
  }
  storematrix[i,2]=out
}
acceptance_rate = sum(storematrix[,1])/num_draws
plot(storematrix[,2])
acceptance_rate

#####################################################################
#rstan
library('rstan')
rstan_options(auto_write = TRUE)

mod = stan_model(file="stdnormal.stan")

fit = sampling(mod,warmup=0,iter=100,chain=1,algorithm="HMC",control=list(metric="unit_e",stepsize_jitter=0,stepsize=1,int_time=10,adapt_engaged=F))
samples=extract(fit,pars="y")
diag=get_sampler_params(fit)
plot(samples[[1]])
