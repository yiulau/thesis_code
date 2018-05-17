import pystan,pickle,numpy,os
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
if os.path.isfile('result_from_long_chain.pkl'):
    print("file already exists")
else:

    fit = mod.sampling(data=data,seed=1,iter=500000,thin=10)

    correct_samples = fit.extract(permuted=True)["beta"]
    #print(fit)
    correct_mean = numpy.mean(correct_samples,axis=0)
    #print(correct_mean)
    correct_cov = numpy.cov(correct_samples,rowvar=False)

    #print(correct_cov)
    out = {"correct_mean":correct_mean,"correct_cov":correct_cov}
    with open('result_from_long_chain.pkl', 'wb') as f:
        pickle.dump(out, f)
