import os.path
import urllib.request
# download data if it does not already exist
# if it exist returns data
import urllib
import urllib.request

# s&p500 data

url = "https://raw.githubusercontent.com/pymc-devs/pymc3/de1b8c839ec4af4e7ca492cdac70abbc38849623/pymc3/examples/data/SP500.csv"

urllib.request.urlretrieve(url, 'sp500.csv')
exit()

# heart data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"

urllib.request.urlretrieve(url, 'heart.csv')


# boston dataset
url = 'http://lib.stat.cmu.edu/datasets/boston'

urllib.request.urlretrieve(url, 'boston.csv')


# german credit

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

urllib.request.urlretrieve(url,"german.csv")