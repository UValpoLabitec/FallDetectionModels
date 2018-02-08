import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model
import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from textInBatch import data_sequence


class CONV_LSTM_model():
	"""
	A convolutional recurrent NN model implementation using LSTM layers
	"""

	def __init__(self, model="", 
				seq_length = 1, l2_lambda = 0.0001, 
				num_epochs = 3000, compile=True):
				
		self.loss = 0
		self.acc = 0
		self.seq_length = seq_length
		self.l2_lambda = l2_lambda
		self.num_epochs = num_epochs
		if model != "":
			self.load(model, compile)
		else:
			self.create_network()
			

	def load(self, file, compile=True):
		try:
			del self.network
		except Exception:
			pass
		json_file = open(file + ".json", "r")	
		loaded_model_json = json_file.read()
		json_file.close()
		self.network = model_from_json(loaded_model_json)
		self.network.load_weights(file + ".h5")
		if compile == True:
			rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
			self.network.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
					
	def save(self, file):
		model_json = self.network.to_json()	
		with open(file + ".json", "w") as json_file:
			json_file.write(model_json)
		#self.network.save_weights(file + ".h5")
	
	def create_network(self):
		self.network = Sequential()
		self.network.add(Conv1D(filters=32, input_shape = (self.seq_length, 32), kernel_size=3, padding='same', kernel_initializer='he_uniform', 
						kernel_regularizer=l2(self.l2_lambda),activation='relu'))
		self.network.add(MaxPooling1D(pool_size=2))
		self.network.add(Dropout(0.2))
		self.network.add(LSTM(100, kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros', 
				unit_forget_bias=True, 
				kernel_regularizer=None, 
				recurrent_regularizer=None, 
				bias_regularizer= None, 
				activity_regularizer= None, 
				kernel_constraint=None, 
				recurrent_constraint=None, 
				bias_constraint=None, 
				return_sequences=True, 
				return_state=False, stateful=False))				
		self.network.add(Dropout(0.2))
		self.network.add(LSTM(100, kernel_initializer='glorot_uniform', 
				recurrent_initializer='orthogonal', 
				bias_initializer='zeros', 
				unit_forget_bias=True, 
				kernel_regularizer=None, 
				recurrent_regularizer=None, 
				bias_regularizer= None, 
				activity_regularizer= None, 
				kernel_constraint=None, 
				recurrent_constraint=None, 
				bias_constraint=None, 
				return_sequences=False, 
				return_state=False, stateful=False))				
		#self.network.add(Dropout(0.2))	
		self.network.add(Dense(1, kernel_initializer="uniform"))
		self.network.add(BatchNormalization())
		self.network.add(Activation('sigmoid'))
		plot_model(self.network, to_file='model.png', show_shapes=True, show_layer_names=True)
		self.compile()
	
	def compile(self):
		sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
		rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
		self.network.compile(loss='binary_crossentropy', optimizer= rmsprop, metrics=['accuracy'])
		print(self.network.summary())
	
	def getData(self, X_train, Y_train, X_test, Y_test):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
        	
	
	def train(self, file):
		tensorboard = TensorBoard(log_dir="logs1/{}".format(time()), histogram_freq=0, batch_size=4, write_graph=True, write_grads=True, 
						write_images=True, embeddings_freq=0, embeddings_layer_names=True, embeddings_metadata=None)
		checkpoint = ModelCheckpoint(file + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
			
		self.network.fit(self.X_train, self.Y_train, batch_size=4,
                        epochs=self.num_epochs,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=500), checkpoint, tensorboard])
		
	def evaluate(self):
		scores = self.network.evaluate(self.X_test, self.Y_test, verbose=1)
		y_pred = self.network.predict_classes(self.X_test,verbose=1, batch_size=1)
		target_names = ['non fall', 'fall']
		print(classification_report(self.Y_test, y_pred, target_names=target_names))
		print("Accuracy: %.2f%%" % (scores[1]*100))
		
		


	
# #read xtrain data
X_train, Y_train = data_sequence('NNData/')

# #read xtest data
X_test, Y_test = data_sequence('testData/') 

	
model = CONV_LSTM_model("conv_lstm_model")
model.getData(X_train, Y_train, X_test, Y_test)
#model.train("conv_lstm_model")
#model.save("conv_lstm_model")
model.evaluate()		
