import os
import pandas as pd
#from distributions.stochastic_volatility.stochastic_volatility import V_stochastic_volatility
from distributions.stochastic_volatility.stochastic_volatility import V_stochastic_volatility
address1 = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data"
address2 = "/home/yiulau/work/thesis_code/explain_hmc/input_data"
address_list = [address1,address2]
for address in address_list:
    if os.path.exists(address):
        abs_address = address + "/sp500.csv"
df = pd.read_csv(abs_address, header=0, sep=" ")
dfm = df.as_matrix()
print(dfm.shape)
print(dfm[:,0])

class V_sp500_stochastic_volatility(V_stochastic_volatility):
    def __init__(self):
        address1 = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data"
        address2 = "/home/yiulau/work/thesis_code/explain_hmc/input_data"
        address_list = [address1, address2]
        for address in address_list:
            if os.path.exists(address):
                abs_address = address + "/sp500.csv"
        df = pd.read_csv(abs_address, header=0, sep=" ")
        dfm = df.as_matrix()
        self.input = dfm[:, 0]
        #print(dfm.shape)
        #print(dfm[:, 0])
        super(V_sp500_stochastic_volatility, self).__init__()

