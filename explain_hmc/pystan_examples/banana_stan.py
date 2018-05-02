import pystan,pickle

code = """
functions { 
  real banana_log(vector x) { 
     real lprob;
     lprob <- -(100*(x[2]-x[1]^2)^2 + (1-x[1])^2)/20;
     return lprob;
  }
} 
parameters{
  vector[2] y;
}
model{
  y ~ banana();
}
"""

recompile = False
if recompile:
    mod = pystan.StanModel(file="./banana.stan")
    with open('model.pkl', 'wb') as f:
        pickle.dump(mod, f)

mod = pickle.load(open('model.pkl', 'rb'))

#fit = pystan.stan(model_code=code,iter=1000,chains=4)

out = mod.sampling()

