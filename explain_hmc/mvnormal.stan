data {
  int<lower=1> N;
  vector[N] mu;
  matrix[N,N] Sigma;
}
parameters {
  vector[N] y;
}
model{
  y ~ multi_normal(mu,Sigma);
}