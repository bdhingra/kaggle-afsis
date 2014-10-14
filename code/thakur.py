# beating the benchmark
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation

train = pd.read_csv('../data/spectra_princomp.csv')
test = pd.read_csv('../data/test_spectra_princomp.csv')
labels = pd.read_csv('../data/target.csv')

xtrain, xtest = np.array(train), np.array(test)

sup_vec = svm.SVR(C=10000.0, verbose = 2)

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)

sample = pd.read_csv('../data/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('beating_benchmark.csv', index = False)
