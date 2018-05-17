library(RcppCNPy)
library(coda)
library(mcmcse)

mcse_matrix = npyLoad("temp_mcse_matrix.npy")

#dim(mcse_matrix)

out = rep(0,dim(mcse_matrix)[2])
for (cur in 1:dim(mcse_matrix)[2]){
  out[cur]=mcse(mcse_matrix[,cur])$se
}
#out = mcse(mcse_matrix)$se
npySave("temp_mcse_matrix_out.npy",out)
