# Python script to perform linear regression on spectral data

import csv
import numpy
from sklearn import linear_model

# Load the principal components and target variables and test data
data = numpy.genfromtxt('../data/spectra_princomp.csv', delimiter=',')
target = numpy.genfromtxt('../data/target.csv', delimiter=',')
test = numpy.genfromtxt('../data/test_spectra_princomp.csv', delimiter=',')
test_pidn = []
with open('../data/test_pidn.csv','rb') as csvfile:
	datareader = csv.reader(csvfile)
	for row in datareader:
		test_pidn.append(row)

# Fit linear model
regr = linear_model.LinearRegression()
regr.fit(data,target)

# Get test predictions
test_prediction = regr.predict(test)
tpr = test_prediction.tolist()

# Save to file with PIDN
with open('../predictions/linear_regression.csv','wb') as csvfile:
	datawriter = csv.writer(csvfile, delimiter=',')
	datawriter.writerow(['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand'])
	for i in range(0,len(test_prediction)):
		datawriter.writerow(test_pidn[i] + tpr[i])
