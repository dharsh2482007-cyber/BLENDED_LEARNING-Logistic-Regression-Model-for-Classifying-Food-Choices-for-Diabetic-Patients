# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset, then separate features (X) and target variable (y).


2.Preprocess the data by scaling features and encoding the target labels.


3.Split the dataset into training and testing sets and train the Logistic Regression model.


4.Predict the class labels and evaluate the model using metrics such as accuracy, precision, recall, and F1-score.


## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as ans
import matplotlib.pyplot as plt

df=pd.read_csv('food_items (1).csv')
print('Name: KRITHIKAA P')
print('Reg. No: 212225040193')
print('Dataset Overview:')
print(df.head())
print("\nDataset Info:")
print(df.info())

X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000

l2_model=LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred=l2_model.predict(X_test)

print('Name: KRITHIKAA P')
print('Reg. No: 212225040193')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print('Name: KRITHIKAA P')
print('Reg. No: 212225040193')
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
