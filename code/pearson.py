# Python script to compute pearson correlation

import csv
import numpy
from scipy.stats import pearsonr

# Load the principal components and target variables and test data
data = numpy.genfromtxt('../data/spectra_kernelprincomp.csv', delimiter=',')
target = numpy.genfromtxt('../data/target.csv', delimiter=',')
test = numpy.genfromtxt('../data/test_spectra_princomp.csv', delimiter=',')
train_loc = numpy.genfromtxt('../data/train_location.csv', delimiter=',')
test_loc = numpy.genfromtxt('../data/test_location.csv', delimiter=',')
test_pidn = []
train_depth = []
test_depth = []
with open('../data/test_pidn.csv','rb') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
                test_pidn.append(row)
with open('../data/train_depth.csv','rb') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
                train_depth.append(float(row[0]))
with open('../data/test_depth.csv','rb') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
                test_depth.append(float(row[0]))

# Split training set based on depth
train1 = []
train2 = []
target1 = []
target2 = []
tloc1 = []
tloc2 = []
for i in range(len(data)):
        if train_depth[i] == 1:
                train1.append(data[i])
                target1.append(target[i])
		tloc1.append(train_loc[i])
        else:
                train2.append(data[i])
                target2.append(target[i])
		tloc2.append(train_loc[i])

#  Split test set based on depth
test1 = []
test2 = []
tpidn1 = []
tpidn2 = []
for i in range(len(test)):
        if test_depth[i] == 1:
                test1.append(test[i])
		tpidn1.append(test_pidn[i])
        else:
                test2.append(test[i])
                tpidn2.append(test_pidn[i])

# Compute pearson coefficients
with open('../data/pearson2.csv','wb') as csvfile:
	datawriter = csv.writer(csvfile)
	# Topsoil
	# spectral predictors
	for i in range(10):
		x = [row[i] for row in train1]
		r = []
		for j in range(5):
			y = [row[j] for row in target1]
			p = pearsonr(x, y)
			r.append(p[0])
		datawriter.writerow([i] + r)

	# location predictors
	for i in range(15):
		x = [row[i] for row in tloc1]
		r = []
		for j in range(5):
			y = [row[j] for row in target1]
			p = pearsonr(x, y)
			r.append(p[0])
		datawriter.writerow([i] + r)

	datawriter.writerow([])
	datawriter.writerow([])
	datawriter.writerow([])

	# Subsoil
	# spectral predictors
	for i in range(10):
		x = [row[i] for row in train2]
		r = []
		for j in range(5):
			y = [row[j] for row in target2]
			p = pearsonr(x, y)
			r.append(p[0])
		datawriter.writerow([i] + r)

	# location predictors
	for i in range(15):
		x = [row[i] for row in tloc2]
		r = []
		for j in range(5):
			y = [row[j] for row in target2]
			p = pearsonr(x, y)
			r.append(p[0])
		datawriter.writerow([i] + r)
