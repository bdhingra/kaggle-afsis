# beating the benchmark
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../data/spectra_princomp.csv', header=None)
test = pd.read_csv('../data/test_spectra_princomp.csv', header=None)
labels = pd.read_csv('../data/target.csv', header=None)

xtrain, xtest, xlabels = np.array(train), np.array(test), np.array(labels)

tuned_parameters = [{'gamma':[1e-5, 1e-3, 0.1, 1, 100, 10000], 'C':[0.001, 0.01, 1, 100, 10000]}]

clf = GridSearchCV(svm.SVR(C=1), tuned_parameters, cv = 10, scoring='mean_squared_error')
clf.fit(xtrain, xlabels)

print("Best params:\n")
print(clf.best_estimator_)

