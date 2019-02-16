# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn

# Helper libraries
import numpy as np
import sys
sys.path.append("..")
import utils

DATA_DIR = "../../Datasets/Final_Data/normalized_all_vectors_merged_timeseries(10)_btc_only.csv"
TRAIN_DIR = "../../Datasets/Final_Data/normalized_all_vectors_merged_timeseries(10)_btc_ltc_eth.csv"
TEST_DIR = "../../Datasets/Final_Data/normalized_all_vectors_merged_timeseries(10)_dash_only.csv"


# RNN parametres
learning_rate = 0.001
epochs = 200
n_classes = 1
n_units = 128
input_length = 10
number_of_sequences = 18
batch_size = 64
num_layers = 2
drop_prob = 0.2
keep_prob = 0.8

xplaceholder= tf.placeholder('float',[None,input_length,number_of_sequences])
yplaceholder = tf.placeholder('float',[None,n_classes])


def multilayer_rnn_model(cell_flag):

	x = tf.unstack(xplaceholder,input_length,axis=1)

	lstm_cells = []
	for _ in range(num_layers):
		if(cell_flag==1):
			cell = tf.nn.rnn_cell.LSTMCell(n_units)
		else:
			cell = tf.nn.rnn_cell.GRUCell(n_units)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-drop_prob)
		lstm_cells.append(cell)

	lstm_layers = rnn.MultiRNNCell(lstm_cells)
	outputs, states = tf.nn.static_rnn(lstm_layers, x, dtype=tf.float32)

	output = outputs[-1]
	w_softmax = tf.Variable(tf.truncated_normal([n_units, n_classes]))
	b_softmax = tf.Variable(tf.random_normal([n_classes]))
	logit = tf.matmul(output, w_softmax) + b_softmax
	dropped_out_logits = tf.nn.dropout(logit, keep_prob)

	return dropped_out_logits

logit = multilayer_rnn_model(1)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yplaceholder))

optim = 1
if(optim==0):
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
else:
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

print("Loading data..")

# train mode=0 train and test for btc --- =1 train with btc and 1,2 other coins -> test for a random coin
# validation mode = 1 --> validation with train set --- validation mode =0 --> validation with the half of test
# and test with the other half

train_mode = 1
validation_mode = 1

if(train_mode==1):
	data = utils.load_dataset(DATA_DIR)
	inputs,labels,feature_num = utils.convert_data_to_arrays(data,1)
	train_inputs,train_labels,test_inputs,test_labels = utils.smash_data_for_timeseries(inputs,labels)

	if(validation_mode == 1):
		valid_inputs = train_inputs
		valid_labels = train_labels
	else:
		#test_inputs,test_labels,valid_inputs,valid_labels = utils.split_test_and_valid(test_inputs,test_labels)
		valid_inputs = test_inputs
		valid_labels = test_labels
else:
	train_data = utils.load_dataset(TRAIN_DIR)
	train_inputs,train_labels,feature_num = utils.convert_data_to_arrays(train_data,1)
	test_data = utils.load_dataset(TEST_DIR)
	test_inputs,test_labels,feature_num = utils.convert_data_to_arrays(test_data,1)
	_,_,test_inputs,test_labels = utils.smash_data_for_timeseries(test_inputs,test_labels)

	if(validation_mode == 1):
		valid_inputs = train_inputs
		valid_labels = train_labels
	else:
		#test_inputs,test_labels,valid_inputs,valid_labels = utils.split_test_and_valid(test_inputs,test_labels)
		valid_inputs = test_inputs
		valid_labels = test_labels

test_asset = 'btc'

train_iters = 0
if(len(train_labels) % batch_size == 0):
    train_iters = len(train_labels) / batch_size
else:
    train_iters = len(train_labels) / batch_size + 1

print("Optimization starting..")

    
with tf.Session() as sess:
	sess.run(init)

	epoch_loss_list = []
	epoch_loss_list2 = []
	epoch_index_list = []

	for epoch in range(epochs):
		epoch_loss = 0

		train_inputs,train_labels =  utils.unison_shuffled_copies(train_inputs,train_labels)

		for i in range(train_iters):

			batch_x = utils.next_batch(train_inputs,i,batch_size)
			batch_y = utils.next_batch(train_labels,i,batch_size)

			batch_x = batch_x.reshape((-1,input_length,number_of_sequences))
			batch_y = batch_y.reshape((-1,n_classes))

			_, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})

			epoch_loss += c/train_iters

		#valid_x = valid_inputs.reshape((-1,input_length,number_of_sequences))
		#valid_y = valid_labels.reshape((-1,n_classes))
		#valid_loss = sess.run([cost], feed_dict={xplaceholder: valid_x, yplaceholder: valid_y})
		valid_loss = 0

		epoch_loss_list.append(epoch_loss)
		epoch_index_list.append(epoch+1)
		epoch_loss_list2.append(valid_loss)

		print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss, ' validation loss: ',valid_loss)

	utils.plot_epoch_loss(epoch_index_list,epoch_loss_list,epoch_loss_list2)


	print("Optimization Finished!")

	#testing

	net_output = tf.round(tf.nn.sigmoid(logit))
	correct_preds = tf.equal(net_output,yplaceholder)
	accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))
	test_inputs = test_inputs.reshape((-1,input_length,number_of_sequences))

	t_labels = test_labels

	test_labels = test_labels.reshape((-1,n_classes))


	print("Tf-Accuracy: ",accuracy.eval({xplaceholder: test_inputs, yplaceholder: test_labels}))
	predictions = correct_preds.eval({xplaceholder: test_inputs, yplaceholder: test_labels})
	utils.figure_faults_timeseries(predictions,test_asset)

	preds = sess.run([net_output], feed_dict = {xplaceholder: test_inputs})
	preds = np.array(preds)[0]

	utils.calculate_metrics(preds,t_labels)
	
