from distributions.response_model import V_response_model
import torch
from finite_differences.finite_diff_funcs import compute_and_display_results
from finite_differences.finite_diff_funcs import compute_and_display_results
response_model_object = V_response_model()

dim = response_model_object.dim
response_model_object.beta.data.copy_(torch.randn(dim))
cur_beta = response_model_object.beta.data.clone()

compute_and_display_results(response_model_object,10)




