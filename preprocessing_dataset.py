import numpy as np
import csv

def get_dataset_in_np(dataset_path):
	dataset = []
	with open(dataset_path) as f:
		dataset_csv_reader = csv.reader(f, delimiter=",")
		for line in dataset_csv_reader:
			temp_line = []

			for word in line:
				temp_line.append(word)
			line = temp_line

			dataset.append(line)

	dataset = np.array(dataset[1:], dtype='float')
	return dataset

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