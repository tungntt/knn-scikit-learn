import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, model_selection, metrics

import os

# Load data
current_path = os.getcwd()
file_path =  current_path + "/iris.data.csv"
data=pd.read_csv(file_path, delimiter=',',header =None,skipinitialspace=True)

# Split data
x_train, x_test,y_train,y_test = train_test_split(data.iloc[:,0:4], data.iloc[:,4],test_size=0.4, random_state=2017)

# Initialize k
k = np.arange(50)+1
params = {'n_neighbors': k}

# Initialize knn
knn = neighbors.KNeighborsClassifier()
clf = model_selection.GridSearchCV(estimator=knn, param_grid=params, cv=10)
clf.fit(X=x_train, y=y_train)

# Get estimator and k chosen
estimator = clf.best_estimator_
n_neighbor = clf.best_params_

# Predict new input
predic_data = estimator.predict(x_test)

score_test = metrics.accuracy_score(y_test, predic_data)
print('Evaluation Score %f' % score_test)
print('K Chosen: %d' % n_neighbor.get('n_neighbors'))



