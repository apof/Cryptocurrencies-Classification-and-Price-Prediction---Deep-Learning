# TensorFlow
import tensorflow as tf
from tensorflow.contrib import rnn

# Helper libraries
import numpy as np
import utils
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

DATA_DIR = "../Datasets/normalized_all_vectors_merged_timeseries.csv"

epochs = 20
n_classes = 1
n_units = 400
n_features = 18*4
batch_size = 32

xplaceholder= tf.placeholder('float',[None,n_features])
yplaceholder = tf.placeholder('float')

def recurrent_neural_network_model():
    layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.split(xplaceholder, n_features, 1)
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
                
			_, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})

			epoch_loss += c/train_iters

		print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

	print("Optimization Finished!")

	correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(yplaceholder, 1))
	# Calculate accuracy
	pred = tf.round(tf.nn.sigmoid(logit)).eval({xplaceholder: np.array(train_inputs), yplaceholder: np.array(train_labels)})
	f1 = f1_score(np.array(train_labels), pred, average='macro')
	accuracy=accuracy_score(np.array(train_labels), pred)
	recall = recall_score(y_true=np.array(train_labels), y_pred= pred)
	precision = precision_score(y_true=np.array(train_labels), y_pred=pred)
	print("F1 Score:", f1)
	print("Accuracy Score:",accuracy)
	print("Recall:", recall)
	print("Precision:", precision)

