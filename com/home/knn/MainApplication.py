import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, model_selection, metrics

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

current_path = os.getcwd()
file_path =  current_path + "/iris.data.csv"
data=pd.read_csv(file_path, delimiter=',',header =None,skipinitialspace=True)
training_x = data.iloc[:,0:4]
x_train, x_test,y_train,y_test = train_test_split(data.iloc[:,0:4], data.iloc[:,4],test_size=0.4, random_state=2017)
# print x_test
# print '-----------------------'
# print data
k = np.arange(50)+1
params = {'n_neighbors': k}
knn = neighbors.KNeighborsClassifier()
clf = model_selection.GridSearchCV(knn, params, cv=10)
clf.fit(X=x_train, y=y_train)
y_pred = clf.predict(x_test)
print('True lable:', np.asarray(y_test[:10]))
print('Predicted lable:',y_pred[:10])

score_test = metrics.accuracy_score(y_test, y_pred)
print(score_test)


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.figure();
