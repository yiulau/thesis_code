
# compare best found by grid search with best found by gpyopt
# both using similar computational resources
# do the grid search first, then compute total computational resources used.
# start bayesian optimization looking at one point at a time. accumulate total computational resources. continue until
# exceeded previous total computational resources
#
# repeat experiment 10-20 times on same model,same data,same integrator,
# plot H,V,T to see if optimal t is roughly half-period , identify optimal t on graph for easy inspection
# see if optimal from gpyopt is systematically better than grid search
# find first point from gpyopt that beats grid search. and identify computational resources used up to that point
# cost of bayesian optimization is insignificant cuz dimension is 2 , maybe 3 and each point is equivalent to many samples <=>
# leapfrogs steps
# sample models:
# logistic (different data)
# hierarchical logistic
# funnel
# 8 schools (ncp)
# multivariate normal
# banana
# integrator
# unit_e hmc (ep,t) windowed_option
# dense_e , diag_e hmc (ep,t) adapting cov, or cov_diag at the same time windowed option
# softabs - diag,outer_product,diag_outer_product static (ep,t,alpha)
# xhmc - delta , unit_e, dense_e, diag_e, softabs (ep,delta) or (ep,delta,alpha)
#
# performance assessement
# objective function esjd/cost= number of gradients or esjd/seconds
#  ess (min, max , median)


# test for sensitivities to objective functions
# change objective functions keep everything the same

#model vs integrator

# test for

# experiment 3 variables (model,integrator,objective function)
# 3 x 3 numpy matrix. storing experiment information. depending on volume store the chain or just the experiment output

