# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn

# Helper libraries
import numpy as np
import sys
sys.path.append("..")
import utils

DATA_DIR = "../../Datasets/Final_Data/regression_data_btc_norm.csv"

# RNN parametres
learning_rate = 0.001
epochs = 1500
output_neurons = 1
n_units = 128
input_length = 7
number_of_sequences = 6
batch_size = 64
num_layers = 1
drop_prob = 0.1

# DNN parametres
n_hidden_1 = 32

xplaceholder= tf.placeholder('float',[None,input_length,number_of_sequences])
yplaceholder = tf.placeholder('float',[None,output_neurons])


def dnn(lstm_output):

	weights = {
    'h1': tf.Variable(tf.random_normal([n_units, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, output_neurons]))
	}

	biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([output_neurons]))
	}

	layer_1 = tf.add(tf.matmul(lstm_output, weights['h1']), biases['b1'])
	out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

	return out_layer


def multilayer_rnn_model():

	x = tf.unstack(xplaceholder,input_length,axis=1)

	lstm_cells = []
	for _ in range(num_layers):
		cell = tf.nn.rnn_cell.LSTMCell(n_units)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-drop_prob)
		lstm_cells.append(cell)

	lstm_layers = rnn.MultiRNNCell(lstm_cells)
	outputs, states = tf.nn.static_rnn(lstm_layers, x, dtype=tf.float32)

	return dnn(outputs[-1])    

logit = multilayer_rnn_model()
cost=tf.reduce_mean(tf.square(logit-yplaceholder))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

print("Loading data..")


data = utils.load_dataset(DATA_DIR)
inputs,labels,feature_num = utils.convert_data_to_arrays_regression(data)
train_inputs,train_labels,test_inputs,test_labels = utils.smash_data_for_timeseries(inputs,labels)

test_asset = 'btc'

train_iters = 0
if(len(train_labels) % batch_size == 0):
    train_iters = len(train_labels) / batch_size
else:
    train_iters = len(train_labels) / batch_size + 1

print("Optimization starting..")

    
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(epochs):
		epoch_loss = 0

		train_inputs,train_labels =  utils.unison_shuffled_copies(train_inputs,train_labels)

		for i in range(train_iters):

			batch_x = utils.next_batch(train_inputs,i,batch_size)
			batch_y = utils.next_batch(train_labels,i,batch_size)

			batch_x = batch_x.reshape((-1,input_length,number_of_sequences))
			batch_y = batch_y.reshape((-1,output_neurons))

			_, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})

			epoch_loss += c/train_iters

		print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)

	print("Optimization Finished!")

	print("Testing..")

	test_batch = test_inputs.reshape((-1,input_length,number_of_sequences))
	predictions = sess.run([logit], feed_dict = {xplaceholder: test_batch})

	print('Total Mae for Test set is: ',utils.compute_error(test_labels,np.array(predictions)[0]))

	preds = np.array(predictions)[0]

	close_labels = []
	close_preds = []
	index_list = []
	for i in range(0,len(preds)):
		close_labels.append(test_labels[i])
		close_preds.append(preds[i])
		index_list.append(i)
	
	utils.print_preds(close_labels,close_preds,index_list)
	
