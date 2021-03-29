import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, r2_score, f1_score, confusion_matrix, precision_score, recall_score
import csv
import statistics

labels = pd.read_csv('labels.csv')
# print(labels)
# print(labels.head())
Y = labels.iloc[:,-1].values
#print(Y)

dataset = pd.read_csv('dataset.csv')
#print(dataset.head())
X = dataset.iloc[:,2:].values
#print(X)
le = LabelEncoder()
for i in range(X.shape[1]):
    X[:,i] = le.fit_transform(X[:,i])
Y = le.fit_transform(Y)

ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.88)
model = svm.SVC(C = 7.21,kernel='rbf')
model.fit(X_train,Y_train)
cvscore = cross_val_score(model,X_train,Y_train,cv = 10)
see = model.predict(X_test)

 
#print(see)

print("Accuracy: ",accuracy_score(Y_test,see)*100,"%")
print("R2 Score: ", r2_score(Y_test,see)*100,"%" )
print("F1 Score: ",f1_score(Y_test,see,average='weighted')*100,"%")
print("Precision: ",precision_score(Y_test,see,average='weighted')*100,"%")
print("Recall: ",recall_score(Y_test,see,average='weighted')*100,"%")
sns.heatmap(confusion_matrix(Y_test,see),annot = True)
# for i in Y_test:
# #     print(i)

data = pd.merge(dataset, labels)
sns.pairplot(data,hue='clster_labels' ,vars=[("Hio..Keratometric."), ("CV_T.4mm."),("SR_H.5mm..1"),("MS.Axis.6mm..2"),("DSI.5mm.")])
# entered the column names as tuples

# #X_test,Y_test,c = np.loadtxt('ex2data1.txt',delimiter=',', unpack=True)
plt.scatter(X_test[:,2],Y_test)
plt.show()

