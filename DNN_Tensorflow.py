# import tensorflow as tf
import numpy as np
import csv

from preprocessing_dataset import get_dataset_in_np, normalize


DATASET_TRAIN_PATH = 'G:/DL/Porto Seguro’s Safe Driver Prediction/data/train/train_small.csv'
DATASET_TEST_PATH = 'G:/DL/Porto Seguro’s Safe Driver Prediction/data/test/test.csv'

dataset_train = get_dataset_in_np(DATASET_TRAIN_PATH)
dataset_train, min_values, max_values = normalize(dataset_train)	# TODO

print(dataset_train.shape)
print(dataset_train[0])

DIM_INPUT = dataset_train.shape[1]
DIM_OUTPUT = 2
NUM_EPOCHS = 100
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, shape=[None, DIM_INPUT])
y = tf.placeholder(tf.float32)

# define the weights with respective shapes
weights = {
	'w1': tf.Variable(tf.random_normal([DIM_INPUT, 128])),
	'w2': tf.Variable(tf.random_normal([128, 256])),
	'w3': tf.Variable(tf.random_normal([256, 512])),
	'w4': tf.Variable(tf.random_normal([512, DIM_OUTPUT]))
}

# define the biases with respective shapes
biases = {
	'b1': tf.Variable(tf.random_normal([128])),
	'b2': tf.Variable(tf.random_normal([256])),
	'b3': tf.Variable(tf.random_normal([512])),
	'b4': tf.Variable(tf.random_normal([DIM_OUTPUT]))
}

# construct the neural network architecture
def deep_neural_network(input):
	layer_1_output = tf.add(tf.matmul(input, weights['w1']), biases['b1'])
	layer_2_output = tf.add(tf.matmul(layer_1_output, weights['w2']), biases['b2'])
	layer_3_output = tf.add(tf.matmul(layer_2_output, weights['w3']), biases['b3'])
	layer_4_output = tf.add(tf.matmul(layer_3_output, weights['w4']), biases['b4'])

	return layer_4_output


