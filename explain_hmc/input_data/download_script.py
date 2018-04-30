import os.path
import urllib.request
# download data if it does not already exist
# if it exist returns data
import urllib
import urllib.request

url = 'http://lib.stat.cmu.edu/datasets/boston'

urllib.request.urlretrieve(url, 'boston.csv')

# do the rest