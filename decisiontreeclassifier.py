import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image  
import pydotplus
import matplotlib.pyplot as plt

x = pd.read_csv("dataset.csv")
x = x[x.columns[2:]]
y = pd.read_csv("labels.csv")
y = y[y.columns[2:]]

x = x.apply(LabelEncoder().fit_transform)
y = y.apply(LabelEncoder().fit_transform)

dtc = tree.DecisionTreeClassifier(max_depth=3)
dtc.fit(x,y)

dot_data = tree.export_graphviz(dtc, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('graph.png')