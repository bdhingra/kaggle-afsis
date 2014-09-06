# Python script to perform PCA on africa soil data

import csv
from sklearn.decomposition import PCA

# Read the data
with open('../data/training.csv','rb') as f:
	reader = csv.reader(f)
	for row in reader:
		print row
		break

