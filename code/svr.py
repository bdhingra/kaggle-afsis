# script to perform support vector regression on spectral data

import csv
import numpy
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from mcrmse import compute_error
import matplotlib.pyplot as plt

# Load the principal components and target variables and test data
data = numpy.genfromtxt('../data/spectra_princomp.csv', delimiter=',')
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
for i in range(len(data)):
        if train_depth[i] == 1:
                train1.append(data[i])
                target1.append(target[i])

	else:
                train2.append(data[i])
                target2.append(target[i])

# Split test set based on depth
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

# Cross validation on topsoil
kf1 = KFold(len(train1), n_folds=10)
error = []
tr = numpy.array(train1)
ta = numpy.array(target1)
for train,test in kf1:
	er = []
	for i in range(5):
		svr = SVR(kernel='rbf', C=1, gamma=0.1)
		prediction = svr.fit(tr[train], [row[i] for row in ta[train]]).predict(tr[test])
		er.append(compute_error(prediction, [row[i] for row in ta[test]]))
	error.append(er)
plt.plot(error)
me = []
for i in range(5):
	me.append(numpy.mean([row[i] for row in error]))
plt.hold
plt.hlines(me)
plt.savefig('error.png')
