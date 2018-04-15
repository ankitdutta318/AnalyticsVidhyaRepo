# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:11:53 2018

@author: ANKIT
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def missing(dataset):
    
    X = dataset.drop('smoking_status', axis=1)
    y = dataset.smoking_status

    # Convert to a pandas dataframe like in your example
    icols = ['id', 'gender_Male', 'gender_Other', 'ever_married_Yes', 'work_type_Never_worked',
             'work_type_Private', 'work_type_Self-employed', 'work_type_children',
             'Residence_type_Urban', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    jcols = ['smoking_status']
    df = pd.concat([pd.DataFrame(X, columns=icols),
                    pd.DataFrame(y, columns=jcols)], axis=1)
    
    notnans = df[jcols].notnull().all(axis=1)
    df_notnans = df[notnans]
    temp = df_notnans.drop('id', axis=1)
    X = (temp[icols[1:]]).iloc[:,0:].values
    y = df_notnans[jcols]
    # Encoding the Dependent Variable
    from sklearn.preprocessing import LabelEncoder
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    df_notnans.smoking_status = labelencoder_y.fit_transform(df_notnans.smoking_status)
    
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
    
    
    # Predict the NaN rows
    df_nans = df.loc[~notnans].copy()
    X_eval = df_nans[icols]
    X_eval.drop(['id'], axis=1, inplace=True)
    # Taking care of missing values
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X_eval.iloc[:, -1:].values)
    X_eval.iloc[:, -1:] = imputer.transform(X_eval.iloc[:, -1:].values)
    # Feature Scaling
    X_eval = sc.fit_transform(X_eval)
    from predict_missing import classifier as model
    df_nans[jcols] = model.predict(X_eval)
    df_nans
    
    final_df = pd.concat([df_nans, df_notnans], axis=0)
    
    return final_df