import pickle,numpy,pystan
from experiments.correctdist_experiments.prototype import check_mean_var
import pandas as pd
address = "/home/yiulau/work/thesis_code/explain_hmc/input_data/pima_india.csv"
df = pd.read_csv(address,header=0,sep=" ")
#print(df)
dfm = df.as_matrix()
#print(dfm)
#print(dfm.shape)
y_np = dfm[:,8]
y_np = y_np.astype(numpy.int64)
X_np = dfm[:,1:8]
dim = X_np.shape[1]
num_ob = X_np.shape[0]
data = dict(y=y_np,X=X_np,N=num_ob,p=dim)

correct = pickle.load(open("result_from_long_chain.pkl", 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()
#print(correct_diag_cov.shape)
#exit()
#print(correct_mean)
#print(correct_cov)

stan_sampling = True
if stan_sampling:
    recompile = False
    if recompile:
        address = "/home/yiulau/work/thesis_code/explain_hmc/stan_code/alt_log_reg.stan"
        mod = pystan.StanModel(file=address)
        with open('model.pkl', 'wb') as f:
            pickle.dump(mod, f)
    else:
        mod = pickle.load(open('model.pkl', 'rb'))

fit = mod.sampling(data=data, seed=20)

mcmc_samples = fit.extract(permuted=True)["beta"]
out = check_mean_var(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)
exit()
out = check_mean_var(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_diag_cov,diag_only=True)
mean_check,diag_cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)

print(diag_cov_check)

print(pc_mean)
print(pc_cov)
