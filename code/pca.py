# Python script to perform PCA on africa soil data

import csv
from sklearn.decomposition import PCA
import numpy as np

# Read the data
data = []
with open('../data/training.csv','rb') as f:
	reader = csv.reader(f)
	next(reader)
	for row in reader:
		row.pop(0)
		for n,i in enumerate(row):
			if i == 'Topsoil':
				row[n] = 2
			elif i == 'Subsoil':
				row[n] = 1
		current = [float(i) for i in row]
		data.append(current)
data = np.array(data)

# Create predictor and target variables
spectra = [row[0:3578] for row in data]
location = [row[3578:3594] for row in data]
Y = [row[3594:3599] for row in data]

# PCA of the spectra
pca = PCA(0.99)
pca.fit(spectra)
print pca.n_components_
