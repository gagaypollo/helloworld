# helloworld
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv as csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

np.random.seed(0)

#load data
#warnings.filterwarnings('ignore')
#pd.set_option('display.max_columns', 1000)

#Read data for analysis
data=pd.read_csv('covtype.csv')

X=data.loc[:,'Elevation':'Soil_Type40']
y=data['Cover_Type']

rem=['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4',
     'Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4',
     'Soil_Type5','Soil_Type6','Soil_Type9','Soil_Type10','Soil_Type11',
     'Soil_Type7','Soil_Type8','Soil_Type14','Soil_Type15',
     'Soil_Type12','Soil_Type32','Soil_Type33','Soil_Type34',
     'Soil_Type18','Soil_Type17','Soil_Type16','Soil_Type13',
     'Soil_Type19','Soil_Type20','Soil_Type29','Soil_Type27','Soil_Type30',
     'Soil_Type31','Soil_Type35','Soil_Type38','Soil_Type39',
     'Soil_Type40','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type26',
     'Soil_Type21','Soil_Type25','Soil_Type28','Soil_Type36','Soil_Type37']

X.drop(rem, axis=1)
#, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,7)
#train_accuracy =np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))
y_pred = np.empty(len(y_test))
dcx = np.empty(len(neighbors))
rmse = np.empty(len(neighbors))
#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
#print(y_pred.shape)
#dcx = accuracy_score(y_test,y_pred)*100
#print('Accuracy: ',dcx)

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    knn.fit(X_train, y_train)
    #Compute accuracy on the training set
#    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the test set
#    test_accuracy[i] = knn.score(X_test, y_test) 
    y_pred = knn.predict(X_test)
    dcx[i] = accuracy_score(y_test,y_pred)
    print('k = ',k)    
    print('Accuracy: ',dcx[i]*100)
    rmse[i] = sqrt(mean_squared_error(y_test, y_pred))
    print('Test RMSE: %.3f' % rmse[i])


#plt.figure(figsize=(10,6))
#plt.title('k-NN Varying number of neighbors')
#plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label='Training accuracy')
#plt.legend()
#plt.xlabel('Number of neighbors')
#plt.ylabel('Accuracy')
