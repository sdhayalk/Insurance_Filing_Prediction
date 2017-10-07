import numpy as np
import csv

def get_dataset_in_np(dataset_path):
	dataset = []
	first_line_flag = True
	with open(dataset_path) as f:
		dataset_csv_reader = csv.reader(f, delimiter=",")
		for line in dataset_csv_reader:
			if first_line_flag:
				first_line_flag = False
				continue

			temp_line = []

			for word in line:
				temp_line.append(word)
			line = temp_line

			dataset.append(line)

	dataset = np.array(dataset)
	dataset_features = np.array(dataset[:, 2:], dtype='float')
	dataset_labels_temp = np.array(dataset[:, 1], dtype='int')

	dataset_labels = []
	for element in dataset_labels_temp:
		temp = [0, 0]
		temp[int(element)] = 1
		dataset_labels.append(temp)

	dataset_labels = np.array(dataset_labels, dtype='int')

	return dataset_features, dataset_labels

def normalize(dataset):	# TODO
	dataset_new = []
	min_values = []
	max_values = []
	for i in range(0, dataset.shape[1]):
		min_values.append(min(dataset[:, i]))
		max_values.append(max(dataset[:, i]))

	for j in range(0, dataset.shape[1]):
		dataset[:, j] = (dataset[:, j] - min_values[j]) / (max_values[j] - min_values[j])
	
	return np.array(dataset), min_values, max_values


def get_test_dataset_in_np(dataset_path):
	dataset = []
	first_line_flag = True
	with open(dataset_path) as f:
		dataset_csv_reader = csv.reader(f, delimiter=",")
		for line in dataset_csv_reader:
			if first_line_flag:
				first_line_flag = False
				continue

			temp_line = []

			for word in line:
				temp_line.append(word)
			line = temp_line

			dataset.append(line)

	dataset = np.array(dataset)
	dataset_features = np.array(dataset[:, 1:], dtype='float')
	dataset_features_id = np.array(dataset[:, 0], dtype='int')

	return dataset_features, dataset_features_id


def normalize_test_dataset(dataset, min_values, max_values):
	dataset_new = []
	
	for j in range(0, dataset.shape[1]):
		dataset[:, j] = (dataset[:, j] - min_values[j]) / (max_values[j] - min_values[j])
	
	return np.array(dataset)
