import tensorflow as tf
import numpy as np
import csv

from preprocessing_dataset import get_dataset_in_np, normalize, get_test_dataset_in_np, normalize_test_dataset

'''this functions returns a batch of features/labels of size BATCH_SIZE
	@
[description]
'''
def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]

DATASET_TRAIN_PATH = 'G:/DL/Porto Seguro’s Safe Driver Prediction/data/train/train.csv'
DATASET_TEST_PATH = 'G:/DL/Porto Seguro’s Safe Driver Prediction/data/test/test.csv'

dataset_features_train, dataset_labels_train = get_dataset_in_np(DATASET_TRAIN_PATH)
dataset_features_train, min_values, max_values = normalize(dataset_features_train)
dataset_features_test, dataset_features_test_id = get_test_dataset_in_np(DATASET_TEST_PATH)
dataset_features_test = normalize_test_dataset(dataset_features_test, min_values, max_values)

# divide dataset into training and validation sets
dataset_features_validation = dataset_features_train[450000:dataset_features_train.shape[0], :]
dataset_labels_validation = dataset_labels_train[450000:dataset_labels_train.shape[0], :]
dataset_features_train = dataset_features_train[0:450000, :]
dataset_labels_train = dataset_labels_train[0:450000, :]

# define some hyperparameters
NUM_EXAMPLES = dataset_features_train.shape[0]
DIM_INPUT = dataset_features_train.shape[1]
DIM_OUTPUT = 2
NUM_EPOCHS = 50
BATCH_SIZE = 45

# placeholders for input and true label output
x = tf.placeholder(tf.float32, shape=[None, DIM_INPUT])
y = tf.placeholder(tf.float32, shape=[None, DIM_OUTPUT])

# define the weights with respective shapes
weights = {
	'w1': tf.Variable(tf.random_normal([DIM_INPUT, 128])),
	'w2': tf.Variable(tf.random_normal([128, 256])),
	'w3': tf.Variable(tf.random_normal([256, 512])),
	'w4': tf.Variable(tf.random_normal([512, 512])),
	'w5': tf.Variable(tf.random_normal([512, DIM_OUTPUT]))
}

# define the biases with respective shapes
biases = {
	'b1': tf.Variable(tf.random_normal([128])),
	'b2': tf.Variable(tf.random_normal([256])),
	'b3': tf.Variable(tf.random_normal([512])),
	'b4': tf.Variable(tf.random_normal([512])),
	'b5': tf.Variable(tf.random_normal([DIM_OUTPUT]))
}

# construct the neural network architecture
def deep_neural_network(input):
	layer_1_output = tf.add(tf.matmul(input, weights['w1']), biases['b1'])
	layer_2_output = tf.add(tf.matmul(layer_1_output, weights['w2']), biases['b2'])
	layer_3_output = tf.add(tf.matmul(layer_2_output, weights['w3']), biases['b3'])
	layer_4_output = tf.add(tf.matmul(layer_3_output, weights['w4']), biases['b4'])
	layer_5_output = tf.add(tf.matmul(layer_4_output, weights['w5']), biases['b5'])

	return layer_5_output


logits = deep_neural_network(x)			# returns the value obtained after feed forward of the neural network
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))	# loss function
optimizer = tf.train.AdamOptimizer()	# optimizer type
training = optimizer.minimize(loss)		# minimize loss using the optimizer

# create a session for training and testing
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		# initialize all global variables, which includes weights and biases

	# training start
	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0

		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			batch_x = get_batch(dataset_features_train, i, BATCH_SIZE)	# get batch of features of size BATCH_SIZE
			batch_y = get_batch(dataset_labels_train, i, BATCH_SIZE)	# get batch of labels of size BATCH_SIZE

			_, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})	# train on the given batch size of features and labels
			total_cost += batch_cost

		print("Epoch:", epoch, "\tCost:", total_cost)

		# predict validation accuracy after every epoch
		y_predicted = tf.nn.softmax(logits)
		correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
		accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
		accuracy_validation = accuracy_function.eval({x:dataset_features_validation, y:dataset_labels_validation})
		print("Validation Accuracy in Epoch ", epoch, ":", accuracy_validation)
	# training end
	
	
	# testing start
	y_predicted = tf.nn.softmax(logits)
	batch_x = get_batch(dataset_features_test, 0, BATCH_SIZE)
	y_probabilities =sess.run(y_predicted, feed_dict={x: batch_x})
	i_= 0
	for i in range(1, int(dataset_features_test.shape[0]/BATCH_SIZE)):
		batch_x = get_batch(dataset_features_test, i, BATCH_SIZE)
		y_probabilities = np.concatenate((y_probabilities, sess.run(y_predicted, feed_dict={x: batch_x})), axis=0)
		i_ = i
	batch_x = get_batch(dataset_features_test, i_+1, BATCH_SIZE)
	y_probabilities = np.concatenate((y_probabilities, sess.run(y_predicted, feed_dict={x: batch_x})), axis=0)
	# testing end

# writing probabilities into a csv file
y_probabilities = np.array(y_probabilities)
with open('run1.csv','w') as file:	
	file.write('id,target')
	file.write('\n')
	for i in range(0, dataset_features_test_id.shape[0]):
		if y_probabilities[i, 1] < 0.0001:
			file.write(str(dataset_features_test_id[i]) + ',' + '0.0')
		else:
			file.write(str(dataset_features_test_id[i]) + ',' + str(y_probabilities[i, 1]))
		file.write('\n')

