data {
    int N; # number of obs
    int p; # dimension of model. num of covariates
    int y[N]; # 0-1 response
    matrix[N,p] X; #design matrix
}
parameters {
    vector[p] beta;
}
transformed parameters{
    vector[N] prob;
    for(i in 1:N){
        prob[i] <- inv_logit(dot_product(X[i,:],beta));
    }
}
model{
    beta ~ normal(0,1);
    for(n in 1:N) {
        y[n] ~ bernoulli(prob[n]);
    }
}
