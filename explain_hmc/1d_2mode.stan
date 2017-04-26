functions{
  real mixture_log(real x){
    real lprob;
    lprob <- log(0.3*exp(-(x-10)^2)+0.7*exp(-(x+5)^2));
    return(lprob);
  }
}
parameters{
  real y;
}
model{
  y~mixture();
}