import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits

# digits 8 x 8 classification (10 classes)
#(1797,64)
#digits = load_digits()
##print(digits.data.shape)
#X = digits["data"]
#y = digits["target"]

#print(X.shape)
#print(y.shape)

#import matplotlib.pyplot as plt
#plt.gray()
#plt.matshow(digits.images[0])
#plt.show()
#exit()
#########################################################################################################
# diabetes regression
# (442,10)
#diabetes = load_diabetes()
#X = diabetes["data"]
#y = diabetes["target"]
#print(X.shape)
#print(y.shape)


# breast cancer logisitic regresssion
# (569,30)
#breast = load_breast_cancer()
#X = breast["data"]
#y = breast["target"]

#print(X.shape)
#print(y.shape)

# boston regression
#boston = load_boston()

#X = boston["data"]
#y = boston["target"]

#print(X.shape)
#print(y.shape)

#print(type(X))

#print(type(y))
#exit()
############################################################################################
# heart data logistic regression
# (270.13) -> (270,25)
#df = pd.read_csv("./heart.csv",header=None,sep=" ")

#print(df.shape)
#y = df[13]
#y = y.as_matrix()
#y = (y-1)*1.0
#print(y)

#df = df.drop(13,1)
#print(df.shape)
#df2 = pd.get_dummies(df,columns=[1,2,5,6,8,10,12])
#dfm = df2.as_matrix()

#print(dfm.shape)
#exit()
###############################################################################################
# german credit logistic regression
#(1000,20) -> (1000,61)
#df = pd.read_csv("./german.csv",header=None,sep=" ")

#y = df[20]
#y = y.as_matrix()
#y = (y - 1)*1.0

#df = df.drop(20,1)
#print(y)
#print(df.shape)

#df2 = pd.get_dummies(df,columns=[0,2,3,5,6,8,9,11,13,14,16,18,19])
#dfm = df2.as_matrix()
##print(dfm.shape)
#exit()
#################################################################################################
# australian data logisitc regression
#(690,14) -> (690,42)
#df = pd.read_csv("./australian.dat",header=None,sep=" ")
#y = df[14]
#print(y)
#df = df.drop(14, 1)
#print(df.shape)
#exit()
# columns selected are categorical . convert to dummy variables
#df2 = pd.get_dummies(df,columns=[0,3,4,5,7,8,10,11])
#print(df2)

#dfm = df2.as_matrix()
#ym = y.as_matrix()
#print(ym)
#print(dfm.shape)
#exit()
#print(dfm[1,])

#####################################################################################################
# pima indian
# (532,7)
df = pd.read_csv("./pima_india.csv",header=0,sep=" ")
#print(df)
dfm = df.as_matrix()

y_np = dfm[:,8]
X_np = dfm[:,1:8]

print(X_np.shape)

