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