import torch, numpy,os
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
import pandas as pd


class V_pima_inidan_logit(V_logistic_regression):
    def __init__(self):
        dim = 10
        num_ob = 20
        y_np = numpy.random.binomial(n=1, p=0.5, size=num_ob)
        X_np = numpy.random.randn(num_ob, dim)
        #print(os.getcwd())
        #exit()
        address = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data/pima_india.csv"
        df = pd.read_csv(address, header=0, sep=" ")
        # print(df)
        dfm = df.as_matrix()
        # print(dfm)
        # print(dfm.shape)
        y_np = dfm[:, 8]
        y_np = y_np.astype(numpy.int64)
        X_np = dfm[:, 1:8]
        input_data = {"X_np":X_np,"y_np":y_np}
        super(V_pima_inidan_logit, self).__init__(input_data)

#vobj = V_pima_inidan_logit()