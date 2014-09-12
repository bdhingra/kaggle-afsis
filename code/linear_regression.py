# Python script to perform linear regression on spectral data

import csv
import numpy
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

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

# Concatenate
#train = numpy.concatenate((data,train_loc), axis=1)
#testd = numpy.concatenate((test,test_loc), axis=1)

# Fit polynomial model 1
poly = PolynomialFeatures(degree=2)
regr1 = linear_model.LinearRegression()
regr1.fit(poly.fit_transform(train1),target1)

# Fit polynomial model 2
regr2 = linear_model.LinearRegression()
regr2.fit(poly.fit_transform(train2),target2)

# Get test predictions
test_prediction1 = regr1.predict(poly.fit_transform(test1))
tpr1 = test_prediction1.tolist()
test_prediction2 = regr2.predict(poly.fit_transform(test2))
tpr2 = test_prediction2.tolist()

# Save to file with PIDN
with open('../predictions/polynomial_regression_based_on_depth_20comps.csv','wb') as csvfile:
	datawriter = csv.writer(csvfile, delimiter=',')
	datawriter.writerow(['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand'])
	for i in range(len(test_prediction1)):
		datawriter.writerow(tpidn1[i] + tpr1[i])
	for i in range(len(test_prediction2)):
		datawriter.writerow(tpidn2[i] + tpr2[i])
