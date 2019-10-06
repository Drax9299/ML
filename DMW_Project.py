import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pydotplus
from IPython.display import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle

print("---------Started---------")
#Importing data

dataset = pd.read_csv("D:/Codes/Python codes/heart.csv")
print("---------Shape---------")
print(dataset.shape)
print("---------Head---------")
print(dataset.head)

#Target the dataset

x = dataset.drop('target',axis=1)
y = dataset['target']

#Splitting dataset

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state = 34)

#Training model
#DecisionTree
clasifier = DecisionTreeClassifier()
clasifier.fit(x_train,y_train)

#KNN
knn_clasifier = KNeighborsClassifier(n_neighbors=20)
knn_clasifier.fit(x_train,y_train)

#SVM
svm_clasifier = SVC(kernel='linear')
svm_clasifier.fit(x_train,y_train)

#predict values

y_pred_decision = clasifier.predict(x_test)
y_pred_knn = knn_clasifier.predict(x_test)
y_pred_svm = svm_clasifier.predict(x_test)


#evaluating model


print("*********___________Results of DecisionTreeClassifier___________*********")
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred_decision))
print("____________")
print(confusion_matrix(y_test,y_pred_decision))
print("____________")
print(classification_report(y_test,y_pred_decision))

print("*********___________Results of KNN___________*********")
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred_knn))
print("____________")
print(confusion_matrix(y_test,y_pred_knn))
print("____________")
print(classification_report(y_test,y_pred_knn))

print("*********___________Results of SVM___________*********")
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred_svm))
print("____________")
print(confusion_matrix(y_test,y_pred_svm))
print("____________")
print(classification_report(y_test,y_pred_svm))


print("---------End---------")

#Save Svm model
pickle.dump(svm_clasifier,open('D:/Codes/Python codes/svm_model_.pkl','wb'))


#Visualize DecisionTree
feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

dot_data = tree.export_graphviz(clasifier,out_file=None,feature_names=feature_cols,class_names=['Yes','No'],filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("Decision tree.png")
