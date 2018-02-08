import glob
from random import shuffle
import pandas as pd
import numpy as np
from keras.preprocessing import sequence

#data file by file			
def data_generator(path):
	read_files = glob.glob(path + "*.txt")
	shuffle(read_files)
	print(len(read_files))
	while 1:
		for f in read_files:
			data = pd.read_csv(f, sep='\*| ', engine='python')
			data = data.dropna(axis=1, how='all')
			data = data.drop(data.columns[[0, 1]], axis=1) 
			data = data.apply(pd.to_numeric, errors='ignore')
			data = data.values
			dataX = data[:,0:-1]
			dataY = data[[2], [-1]]
			dataX = dataX.reshape(1, dataX.shape[0], dataX.shape[1])
			yield dataX, dataY
			


def data_sequence(path):
	"""
	Read sequential data from a the dataset dirs taking
	just the first 60 steps to maintain a fixed sequence 
	length
	"""
	read_files = glob.glob(path + "*.txt")
	shuffle(read_files)
	seqX = []
	seqY = []
	for f in read_files:
		data = pd.read_csv(f, sep='\*| ', engine='python')
		data = data.dropna(axis=1, how='all')
		data = data.drop(data.columns[[0, 1]], axis=1) 
		data = data.apply(pd.to_numeric, errors='ignore')
		data = data.values
		dataX = data[:,0:-1]
		dataY = data[[2], [-1]]
		#dataX = dataX.reshape(1, dataX.shape[0], dataX.shape[1])
		seqX.append(dataX)
		seqY.append(dataY)
	x = sequence.pad_sequences(seqX, maxlen=60, dtype='float32')
	y = np.asarray(seqY)
	return x, y		


 