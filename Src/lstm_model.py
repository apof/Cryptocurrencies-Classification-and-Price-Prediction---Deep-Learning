# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn

# Helper libraries
import numpy as np
import utils
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

DATA_DIR = "../Datasets/Final_Data/normalized_all_vectors_merged_timeseries.csv"

epochs = 500
n_classes = 1
n_units = 300
input_length = 4
number_of_sequences = 18
batch_size = 64

xplaceholder= tf.placeholder('float',[None,input_length,number_of_sequences])
yplaceholder = tf.placeholder('float')

def recurrent_neural_network_model():
    layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.unstack(xplaceholder,input_length,axis=1)
    lstm_cell = rnn.BasicLSTMCell(n_units)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']
    return output

logit = recurrent_neural_network_model()
logit = tf.reshape(logit, [-1])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yplaceholder))
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

print("Loading data..")
	
data = utils.load_dataset(DATA_DIR)

train,test = utils.smash_train_test(data)

train_inputs,train_labels,feature_num = utils.convert_data_to_arrays(train,1)
test_inputs,test_labels,feature_num = utils.convert_data_to_arrays(test,1)

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

		for i in range(train_iters):

			batch_x = utils.next_batch(train_inputs,i,batch_size)
			batch_y = utils.next_batch(train_labels,i,batch_size)

			batch_x = batch_x.reshape((-1,input_length,number_of_sequences))

			_, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})

			epoch_loss += c/train_iters

		print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

	print("Optimization Finished!")

	correct_preds = tf.round(tf.nn.sigmoid(logit)) 

	accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"))

	# Calculate accuracy
	test_inputs = test_inputs.reshape((-1,input_length,number_of_sequences))
	print("Accuracy: ",accuracy.eval({xplaceholder: np.array(test_inputs), yplaceholder: np.array(test_labels)}))