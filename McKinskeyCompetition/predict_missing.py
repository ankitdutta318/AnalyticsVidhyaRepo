# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:32:15 2018

@author: ANKIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/train.csv')
temp = dataset.stroke
dataset.drop('stroke', axis=1, inplace=True)
dataset = pd.concat([temp, dataset], axis=1)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical variable
dataset = pd.get_dummies(dataset, 
                         columns=['gender', 'ever_married', 'work_type', 'Residence_type'], 
                         drop_first=True, dummy_na=False)
X = dataset.drop(['smoking_status'], axis=1)

# Convert to a pandas dataframe like in your example
icols = ['stroke', 'id', 'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked',
         'work_type_Private', 'work_type_Self-employed', 'work_type_children',
         'Residence_type_Urban', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
jcols = ['smoking_status']
df = pd.concat([pd.DataFrame(X, columns=icols),
                pd.DataFrame(y, columns=jcols)], axis=1)

        
notnans = df[jcols].notnull().all(axis=1)
df_notnans = df[notnans]
df_notnans.drop('id', axis=1, inplace=True)
X = (df_notnans[icols[2:]]).iloc[:,0:].values
y = df_notnans[jcols]
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 12:])
X[:, 12:] = imputer.transform(X[:, 12:])

# Split into 75% train and 25% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=4)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Predict the NaN rows
df_nans = df.loc[~notnans].copy()
X_eval = df_nans[icols]
X_eval.drop(['stroke', 'id'], axis=1, inplace=True)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_eval.iloc[:, -1:].values)
X_eval.iloc[:, -1:] = imputer.transform(X_eval.iloc[:, -1:].values)
# Feature Scaling
X_eval = sc.fit_transform(X_eval)
df_nans[jcols] = classifier.predict(X_eval)
df_nans
