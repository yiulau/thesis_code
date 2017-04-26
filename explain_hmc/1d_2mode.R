library('rstan')
rstan_options(auto_write = TRUE)

mod = stan_model(file="1d_2mode.stan")
fit = sampling(mod,iter=1000,init=list(list(y=c(0))),chains=1)
out=extract(fit,pars="y")
plot(out[[1]])

###########################################################
# HMC sampler implemented in R
target_log_lik = function(x){
  -log(0.3*exp(-(x-10)^2)+0.7*exp(-(x-1)^2))
}
target_log_grad = function(x){
  out = -0.3*(x-10)*2*exp(-(x-10)^2)-0.7*(x-1)*2*exp(-(x-1)^2)
  out = out / (0.3*exp(-(x-10)^2)+0.7*exp(-(x-1)^2))
  out = -out
  return(out)
}
# Start sampling
num_draws = 5000
epsilon = 1
L = 16
initq = 1.5
alph=1.03
storematrix = array(0,dim=c(num_draws,2))
storematrix[1,2]=initq

for(i in 2:num_draws){
  out = tempered_HMC(U=target_log_lik,gradU=target_log_grad,epsilon=epsilon,L=L,currentq=storematrix[i-1,2],alpha=alph)
  if(out!=storematrix[i-1,2]){
    storematrix[i,1]=1
  }
  storematrix[i,2]=out
}
acceptance_rate = sum(storematrix[,1])/num_draws
plot(storematrix[,2])
acceptance_rate
####################################################################

tempered_HMC = function(U,gradU,epsilon,L,currentq,alpha){
  q = currentq
  p = rnorm(length(q),0,1)
  currentp = p
  
  for (i in 1:(L)/2){
    p = p * sqrt(alpha)
    p = p - epsilon * gradU(q)/2
    q = q + epsilon * p
    p = p - epsilon * gradU(q)/2
    p = p * sqrt(alpha)
  }
  for(i in ((L/2)+1):L){
    p = p / sqrt(alpha)
    p = p - epsilon * gradU(q)/2
    q = q + epsilon * p
    p = p - epsilon * gradU(q)/2
    p = p / sqrt(alpha)
  }
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
