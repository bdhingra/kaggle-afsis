# beating the benchmark
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../data/spectra_princomp.csv', header=None)
test = pd.read_csv('../data/test_spectra_princomp.csv', header=None)
labels = pd.read_csv('../data/target.csv', header=None)

xtrain, xtest, xlabels = np.array(train), np.array(test), np.array(labels)

tuned_parameters = [{'gamma':[0.005, 0.01, 0.05, 0.1, 0.5], 'C':[0.05, 0.1, 0.5, 1, 5, 10, 50]}]

clf = GridSearchCV(svm.SVR(C=...), tuned_parameters, cv = 10, scoring='mean_squared_error', verbose=1)
clf.fit(xtrain, xlabels[:,1])

print("Best params:\n")
print(clf.best_estimator_)

print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))
print()
