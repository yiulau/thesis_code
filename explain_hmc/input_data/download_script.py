import os.path
import urllib.request
# download data if it does not already exist
# if it exist returns data
import urllib
import urllib.request


# heart data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"

urllib.request.urlretrieve(url, 'heart.csv')


# boston dataset
url = 'http://lib.stat.cmu.edu/datasets/boston'

urllib.request.urlretrieve(url, 'boston.csv')


# german credit

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

urllib.request.urlretrieve(url,"german.csv")