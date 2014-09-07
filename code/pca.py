# Python script to perform PCA on africa soil data

import csv
from sklearn.decomposition import PCA
import numpy as np
from sklearn import linear_model

# Read the data
train_spectra = []
train_location = []
train_Y = []
test_pidn = []
test_spectra = []
test_location = []
with open('../data/training.csv','rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	next(datareader)
	for row in datareader:
		train_spectra.append([float(i) for i in row[1:3579]])
		train_location.append([i for i in row[3579:3595]])
		train_Y.append([float(i) for i in row[3595:]])
with open('../data/sorted_test.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',')
	next(datareader)
	for row in datareader:
		test_pidn.append(row[0])
		test_spectra.append([float(i) for i in row[1:3579]])
		test_location.append([i for i in row[3579:]])

# Create predictor and target variables
tr_spectra = np.array(train_spectra)
tr_Y = np.array(train_Y)
te_spectra = np.array(test_spectra)

# PCA on training spectra
pca = PCA(0.99)
pca.fit(tr_spectra)
print pca.n_components_

# Write csv file
with open('../data/test_pidn.csv','wb') as csvfile:
	datawriter = csv.writer(csvfile, delimiter=',')
	for row in test_pidn:
		datawriter.writerow([row])
np.savetxt('../data/spectra_princomp.csv', pca.transform(tr_spectra), delimiter=',')
np.savetxt('../data/target.csv', tr_Y, delimiter=',')
np.savetxt('../data/test_spectra_princomp.csv', pca.transform(te_spectra), delimiter=',')
