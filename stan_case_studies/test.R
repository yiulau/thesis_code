# 
library(rstan)
model= "
data{
  int<lower=1> p;
  vector[p] mu;
  cov_matrix[p] Sigma;
}
parameters{
  vector[p] beta;
}
model{
  beta ~ multi_normal(mu,Sigma);
}
"
m = c(0,0)
Sig = diag(2)
data = list(mu=m,Sigma=Sig,p=length(m))
out = stan(model_code=model,data=data,iter=1000,chains=4)
check_divergences(out)
out = stan(model_code=model,data=data,iter=5000,chains=4,
           algorithm="HMC",control=list(adapt_engaged=FALSE,
                                        stepsize=0.1,int_time=10))

# different scale
Sig = diag(c(10000,1))
data = list(mu=m,Sigma=Sig,p=length(m))
out = stan(model_code=model,data=data,iter=1000,chains=4,
           control=list(metric="unit_e"))

# Highly correlated
Sig = matrix(c(1,0.9,0.9,1),nrow=2)
data = list(mu=m,Sigma=Sig,p=length(m))
out = stan(model_code=model,data=data,iter=10000,chains=4,
           control=list(metric="unit_e"))

