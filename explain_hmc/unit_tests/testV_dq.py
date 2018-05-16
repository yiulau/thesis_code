from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit

v_obj = V_pima_inidan_logit()

q = v_obj.q_point.point_clone()

out1 = v_obj.dq(q.flattened_tensor)

out2 = v_obj.getdV_tensor(q)

print(out1)
print(out2)

print((out1-out2).sum())