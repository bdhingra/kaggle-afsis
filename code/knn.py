# script to perform support vector regression on spectral data

import csv
import numpy
from sklearn.neighbors import KNeighborsRegressor
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
trloc1 = []
trloc2 = []
for i in range(len(data)):
        if train_depth[i] == 1:
                train1.append(data[i])
                target1.append(target[i])
		trloc1.append(train_loc[i])
	else:
                train2.append(data[i])
                target2.append(target[i])
		trloc2.append(train_loc[i])

# Split test set based on depth
test1 = []
test2 = []
tpidn1 = []
tpidn2 = []
teloc1 = []
teloc2 = []
for i in range(len(test)):
        if test_depth[i] == 1:
                test1.append(test[i])
                tpidn1.append(test_pidn[i])
		teloc1.append(test_loc[i])
        else:
                test2.append(test[i])
                tpidn2.append(test_pidn[i])
		teloc2.append(test_loc[i])

# Cross validation on topsoil
kf1 = KFold(len(train1), n_folds=10)
error = []
tr = numpy.array(train1)
ta = numpy.array(target1)
tl = numpy.array(trloc1)
tt = numpy.concatenate((tr, tl), axis=1)
for train,test in kf1:
	er = []
	for i in range(5):
		neigh = KNeighborsRegressor(n_neighbors=25, weights='distance')
		prediction = neigh.fit(tr[train], [row[i] for row in ta[train]]).predict(tr[test])
		er.append(compute_error(prediction, [row[i] for row in ta[test]]))
	error.append(er)
plt.plot(error)
plt.savefig('error1.png')
me = []
for i in range(5):
	me.append(numpy.mean([row[i] for row in error]))
print me

# Cross validation on subsoil
kf2 = KFold(len(train2), n_folds=10)
error = []
tr = numpy.array(train2)
ta = numpy.array(target2)
tl = numpy.array(trloc2)
tt = numpy.concatenate((tr, tl), axis=1)
for train,test in kf2:
	er = []
	for i in range(5):
		neigh = KNeighborsRegressor(n_neighbors=25, weights='distance')
		prediction = neigh.fit(tr[train], [row[i] for row in ta[train]]).predict(tr[test])
		er.append(compute_error(prediction, [row[i] for row in ta[test]]))
	error.append(er)
plt.plot(error)
plt.savefig('error2.png')
me = []
for i in range(5):
	me.append(numpy.mean([row[i] for row in error]))
print me

# Final model and predictions
tpr1 = []
tpr2 = []
for i in range(5):
	neigh1 = KNeighborsRegressor(n_neighbors=10, weights='distance')
	neigh1.fit(train1, [row[i] for row in target1])
	tpr1.append(neigh1.predict(test1))
	neigh2 = KNeighborsRegressor(n_neighbors=5)
	neigh2.fit(train2, [row[i] for row in target2])
	tpr2.append(neigh2.predict(test2))
tp1 = numpy.asarray(tpr1).transpose().tolist()
tp2 = numpy.asarray(tpr2).transpose().tolist()

# Save test outputs
# Save to file with PIDN
with open('../predictions/knn.csv','wb') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',')
        datawriter.writerow(['PIDN', 'Ca', 'P', 'pH', 'SOC', 'Sand'])
        for i in range(len(tp1)):
                datawriter.writerow(tpidn1[i] + tp1[i])
        for i in range(len(tp2)):
                datawriter.writerow(tpidn2[i] + tp2[i])
