# beating the benchmark
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../data/training.csv')
trloc = pd.read_csv('../data/train_location.csv', header=None)
test = pd.read_csv('../data/sorted_test.csv')
labels = pd.read_csv('../data/target.csv', header=None)

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

xtrain, xtest, xlabels, xloc = np.array(train)[:,:3578], np.array(test)[:,:3578], np.array(labels), np.array(trloc)
X = np.concatenate((xtrain, xloc), axis=1)

#tuned_parameters = [{'kernel':['rbf'],'gamma':[0.0005, 0.001, 0.003, 0.005, 0.007, 0.01], 'C':[50, 1000]}]
#
#clf = GridSearchCV(svm.SVR(C=10000), tuned_parameters, cv = 10, scoring='mean_squared_error', verbose=1, n_jobs=4)
#clf.fit(xtrain, xlabels[:,0])
#
#print("Best params:\n")
#print(clf.best_estimator_)
#
#print("Grid scores on development set:")
#print()
#for params, mean_score, scores in clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
#print()

mdl = svm.SVR(C=1000, gamma=0.0005, verbose=1)
mdl.fit(xtrain, xlabels[:,0])
preds = mdl.predict(xtest).astype(float)

sample = pd.read_csv('../data/sub_28sept_v3.csv')
sample['Ca'] = preds

sample.to_csv('../data/sub_17oct_Ca_optimized_v2.csv', index=False)

