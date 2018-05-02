library(rstan)
dim = 5 
num_obs = 10
X = matrix(rnorm(dim*num_obs),num_obs,dim)
y = rbinom(n=num_obs,size=1,prob=0.5)
model = stan_model(file="./alt_log_reg.stan")
o = sampling(model,data=list(y=y,X=X,N=num_obs,p=dim))
print(o)
