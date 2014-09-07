# Python script to perform linear regression on spectral data

import csv
import numpy
from sklearn import linear_model

# Load the principal components and target variables and test data
data = numpy.genfromtxt('../data/spectra_princomp.csv', delimiter=',')
target = numpy.genfromtxt('../data/target.csv', delimiter=',')
test = numpy.genfromtxt('../data/test_spectra_princomp.csv', delimiter=',')
train_loc = numpy.genfromtxt('../data/train_location.csv', delimiter=',')
test_loc = numpy.genfromtxt('../data/test_location.csv', delimiter=',')
test_pidn = []
with open('../data/test_pidn.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	for row in datareader:
		test_pidn.append(row)

# Concatenate
train = numpy.concatenate((data,train_loc), axis=1)
testd = numpy.concatenate((test,test_loc), axis=1)

# Fit linear model
regr = linear_model.LinearRegression()
regr.fit(train,target)

# Get test predictions
test_prediction = regr.predict(testd)
tpr = test_prediction.tolist()

# Save to file with PIDN
with open('../predictions/linear_regression_with_loc.csv','wb') as csvfile:
	datawriter = csv.writer(csvfile, delimiter=',')
	datawriter.writerow(['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand'])
	for i in range(0,len(test_prediction)):
		datawriter.writerow(test_pidn[i] + tpr[i])
