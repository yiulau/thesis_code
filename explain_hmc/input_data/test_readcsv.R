data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")

datam = data.matrix(data)

# Pima indian 
library(MASS)
X1 = Pima.tr
X = Pima.te
X = rbind(X,X1)

data = model.matrix(type~.,X)
data[,2:8] = scale(data[,2:8])
colMeans(data)
y = as.numeric(X$type)-1
df = data.frame(data,y)
write.table(df,file="pima_india.csv")
# Australian
#download
data = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
# preprocessing
# convert categorical var to factors (dummy variables)
data$V1 = as.factor(data$V1)
data$V4= as.factor(data$V4)
data$V5 = as.factor(data$V5)
data$V6 = as.factor(data$V6)
data$V8 = as.factor(data$V8)
data$V9 = as.factor(data$V9)
data$V11 = as.factor(data$V11)
data$V12 = as.factor(data$V12)

datam = model.matrix(V15~.,data=data)

# German credit
data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

datam = model.matrix(V21~.,data=data)
y = data$V21 -1


# Heart disease
heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "thalach","exang", "oldpeak","slope", "ca", "thal", "num")
heart.data$num[heart.data$num>0]=1
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")
for (i in 1:length(chclass)){
  if(chclass[i]=="factor")
  heart.data[,i] = as.factor(heart.data[,i]) 
}
datam = model.matrix(num~.,heart.data)
y = as.numeric(heart.data$num)-1
