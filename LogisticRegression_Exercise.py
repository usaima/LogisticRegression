# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:03:45 2017

@author: OPS
"""
import numpy as np
from sklearn import datasets
from sklearn import linear_model
#import matplotlib.pyplot as plt


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)
#to split data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

logistic = linear_model.LogisticRegression(fit_intercept=True , C=1e5)
logistic.fit(iris_X_train, iris_y_train)

print (logistic.intercept_, logistic.coef_)

"""
[  0.90253316   6.98000716 -38.13251869] [[  1.51252244   4.93369151  -7.81700689  -3.83849959]
 [ -0.0947089   -2.86680472   1.24142075  -2.80757226]
 [ -3.54028238  -3.67685768   8.8707001   16.40375769]]
"""
