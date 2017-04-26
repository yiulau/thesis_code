import pystan

code = """
parameters {
    real y;
}
model {
    y ~ normal(0,1);
}

"""

fit = pystan.stan(model_code=code,iter=100,chains=4)

fit.plot()
