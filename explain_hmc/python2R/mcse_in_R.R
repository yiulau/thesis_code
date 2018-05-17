library(RcppCNPy)
library(coda)
library(mcmcse)

mcse_vec = npyLoad("temp_mcse.npy")

out = mcse(mcse_vec)$se

npySave("temp_mcse_out.npy",out)
