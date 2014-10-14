# Python script to perform PCA on africa soil data

import csv
from sklearn.decomposition import KernelPCA
import numpy as np

# Read the data
train_spectra = []
train_location = []
train_Y = []
train_depth = []
test_pidn = []
test_spectra = []
test_location = []
test_depth = []
with open('../data/training.csv','rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	next(datareader)
	for row in datareader:
		train_spectra.append([float(i) for i in row[1:3579]])
		train_location.append([float(i) for i in row[3579:3594]])
		train_Y.append([float(i) for i in row[3595:]])
		if row[3594] == 'Topsoil':
			train_depth.append(1)
		else:
			train_depth.append(2)
with open('../data/sorted_test.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	next(datareader)
	for row in datareader:
		test_pidn.append(row[0])
		test_spectra.append([float(i) for i in row[1:3579]])
		test_location.append([float(i) for i in row[3579:3594]])
		if row[3594] == 'Topsoil':
			test_depth.append(1)
		else:
			test_depth.append(2)

# Create predictor and target variables
tr_spectra = np.array(train_spectra)
tr_Y = np.array(train_Y)
te_spectra = np.array(test_spectra)

# kernel PCA on training spectra
pca = KernelPCA(kernel='rbf')
XX = pca.fit_transform(tr_spectra)
X = XX[:,0:10]
print len(X)
print len(X[0])

# Write csv file
np.savetxt('../data/spectra_kernelprincomp.csv', X, delimiter=',')
