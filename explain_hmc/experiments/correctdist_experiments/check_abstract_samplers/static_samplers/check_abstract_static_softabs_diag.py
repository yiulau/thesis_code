from abstract.mcmc_sampler import mcmc_sampler,mcmc_sampler_settings_dict
from adapt_util.tune_param_classes.tune_param_setting_util import *
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
from experiments.experiment_obj import tuneinput_class
import pickle
from experiments.correctdist_experiments.prototype import check_mean_var

mcmc_meta = mcmc_sampler_settings_dict(mcmc_id=0,samples_per_chain=500,num_chains=1,num_cpu=1,thin=1,tune_l_per_chain=0,
                                   warmup_per_chain=100,is_float=False,isstore_to_disk=False)

input_dict = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"alpha":[1e6],"second_order":[True],
              "evolve_L":[10],"metric_name":["softabs_diag"],"dynamic":[False],"windowed":[False],"criterion":[None]}
input_dict2 = {"v_fun":[V_pima_inidan_logit],"epsilon":[0.1],"alpha":[1e6],"second_order":[True],
              "evolve_L":[10],"metric_name":["softabs_diag"],"dynamic":[False],"windowed":[True],"criterion":[None]}


tune_settings_dict = tuning_settings([],[],[],[])

tune_dict  = tuneinput_class(input_dict).singleton_tune_dict()
tune_dict2 = tuneinput_class(input_dict2).singleton_tune_dict()

sampler1 = mcmc_sampler(tune_dict=tune_dict,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)
sampler2 = mcmc_sampler(tune_dict=tune_dict2,mcmc_settings_dict=mcmc_meta,tune_settings_dict=tune_settings_dict)

out = sampler1.start_sampling()

mcmc_samples = sampler1.get_samples(permuted=True)
correct = pickle.load(open("result_from_long_chain.pkl", 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()

out = check_mean_var(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)


out = sampler2.start_sampling()

mcmc_samples = sampler2.get_samples(permuted=True)
correct = pickle.load(open("result_from_long_chain.pkl", 'rb'))
correct_mean = correct["correct_mean"]
correct_cov = correct["correct_cov"]
correct_diag_cov = correct_cov.diagonal()

out = check_mean_var(mcmc_samples=mcmc_samples,correct_mean=correct_mean,correct_cov=correct_cov,diag_only=False)
mean_check,cov_check = out["mcmc_mean"],out["mcmc_Cov"]
pc_mean,pc_cov = out["pc_of_mean"],out["pc_of_cov"]
print(mean_check)
print(cov_check)
print(pc_mean)
print(pc_cov)