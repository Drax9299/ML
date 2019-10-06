import pickle
import numpy as n

svm = pickle.load(open('D:/Codes/Python codes/svm_model_.pkl','rb'))
play = 1
while play!=0:
    sample = []
    sample.append(int(input("Enter Age : ")))
    sample.append(int(input("Enter Sex : ")))
    sample.append(int(input("Enter Chest Pain Type : ")))
    sample.append(int(input("Enter Blood Pressure : ")))
    sample.append(int(input("Enter Cholestrol : ")))
    sample.append(int(input("Enter blood sugar : ")))
    sample.append(int(input("Enter ecg results : ")))
    sample.append(int(input("Enter max heart rate : ")))
    sample.append(int(input("Enter Angina induced  : ")))
    sample.append(float(input("Enter oldpeak : ")))
    sample.append(int(input("Enter slope of peak excercise : ")))
    sample.append(int(input("Enter number of major vessels : ")))
    sample.append(int(input("Enter Thalassemia  : ")))

    example = n.array([sample])
    example = example.reshape(len(example),-1)
    result = svm.predict(example)
    if result == 1:
        print("Chance of Heart Disease")
    else :
        print("No chance of Heart Disease")

    play = int(input("Want to try again? (1/0)"))
