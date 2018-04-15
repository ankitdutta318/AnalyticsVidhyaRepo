# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/new_train.csv')

# Encoding categorical data
# Encoding the Independent Variable
'''dataset = pd.get_dummies(dataset, 
                         columns=['gender', 'ever_married', 'work_type', 'Residence_type'], 
                         drop_first=True, dummy_na=False)'''
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 12:13])
X[:, 12:13] = imputer.transform(X[:, 12:13])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
model = SVC(kernel = 'rbf', random_state = 0)
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#============================= Final Prediction on Tese Set =============================================
from func_pred import missing
testset = pd.read_csv('data/test.csv')
testset = pd.get_dummies(testset, 
                         columns=['gender', 'ever_married', 'work_type', 'Residence_type'], 
                         drop_first=True, dummy_na=False)
df = missing(testset)
y_predset = model.predict(final_df.drop('id', axis=1))