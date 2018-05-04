import pandas as pd

df = pd.read_csv("./australian.dat",header=None,sep=" ")
y = df[14]
#print(y)
df = df.drop(14, 1)
#print(df)
df2 = pd.get_dummies(df,columns=[0,3,4,5,7,8,10,11])
#print(df2)

dfm = df2.as_matrix()
ym = y.as_matrix()
#print(ym)
#print(dfm.shape)
#print(dfm[1,])

df = pd.read_csv("./pima_india.csv",header=0,sep=" ")
print(df)
dfm = df.as_matrix()
print(dfm)