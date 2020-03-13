import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#print(cancer.keys())
#print(cancer['DESCR'])
#print(cancer['target'])
#print(cancer['target_names'])
#print(cancer['feature_names'])
#print(cancer['data'])
#print(cancer['filename'])

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

## Visualizing the Data

#sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

#sns.countplot(df_cancer['target'])

#sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

'''
plt.figure(figsize = (30,30))
sns.heatmap(df_cancer.corr(), annot = True)
'''
#plt.show()

## Model Training
x = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(x_train, y_train)

## Model Evaluation

y_pred = svc_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
#plt.show()


## Improving the MOdel
min_train = x_train.min()
range_train = (x_train - min_train).max()
x_train_scaled =  (x_train - min_train)/range_train

min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_scaled =  (x_test - min_test)/range_test

svc_model.fit(x_train_scaled, y_train)

y_pred = svc_model.predict(x_test_scaled)
cm = confusion_matrix(y_test, y_pred)
#print(cm)
sns.heatmap(cm, annot = True)
#plt.show()

#print(classification_report(y_test, y_pred))

param_grid = {'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(x_train_scaled, y_train)

grid_pred = grid.predict(x_test_scaled)
cm = confusion_matrix(y_test, grid_pred)
print(cm)
sns.heatmap(cm, annot = True)
plt.show()
print(classification_report(y_test, y_pred))
