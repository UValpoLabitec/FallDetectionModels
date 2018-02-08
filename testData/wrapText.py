import glob
from random import shuffle
import pandas as pd
import re

read_files = glob.glob("*.txt")
shuffle(read_files)

# with open("result.txt", "wb") as outfile:
    # for f in read_files:
        # with open(f, "rb") as infile:
            # outfile.write(infile.read())
			
			
			
def data_generator():
	read_files = glob.glob("*.txt")
	shuffle(read_files)
	for f in read_files:
		data = pd.read_csv(f, sep='\*| ', engine='python')
		data = data.dropna(axis=1, how='all')
		data = data.drop(data.columns[[0, 1, 3]], axis=1) 
		data = data.apply(pd.to_numeric, errors='ignore')
		data = data.values
		dataX = data[:,0:-1]
		dataY = data[:, [-1]]
		#X = X.apply(pd.to_numeric, errors='ignore')
		#X = X.values

		#Y = Y.apply(pd.to_numeric, errors='ignore')
		#Y = Y.value
		yield dataX

for item in data_generator():
    print(item)  
# a = data_generator() 
# data = pd.read_csv(read_files[1], sep=' ', engine='python')	
# print(data)