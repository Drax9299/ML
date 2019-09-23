import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib as m

print("---------Started---------")
#Importing data

dataset = pd.read_csv("heart.csv")
print("---------Shape---------")
print(dataset.shape)
print("---------Head---------")
print(dataset.head)

#Target the dataset

x = dataset.drop('target',axis=1)
y = dataset['target']

#Splitting dataset

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)

#Training model

clasifier = DecisionTreeClassifier()
clasifier.fit(x_train,y_train)

#predict values

y_pred = clasifier.predict(x_test)

#evaluating model

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



print("---------End---------")
