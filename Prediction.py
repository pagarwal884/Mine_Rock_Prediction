#Importing the Dependencies

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Data Coollection and Data Processing

sonar_data = pd.read_csv('Mine_Rock_Prediction//sonar data.csv' , header=None)

# number of rows and columns

print(sonar_data.shape)
print(sonar_data.describe()) #describe ---> statistical measures of the data

print(sonar_data[60].value_counts()) #Count the numbers of mines and rocks in the dataset

# M --> Mine
# R --> Rock

print(sonar_data.groupby(60).mean()) # Grouping of data According to rock and mine

# Seprating data and labels
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]

# Divide this data into trining and test data
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.1 , random_state=2 , stratify=Y)

# Data Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Model Training --> Logistic Model

model = LogisticRegression(class_weight='balanced')

#training the logistic regression model using the training data
model.fit(X_train , Y_train)

#Model Evaluation

#Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)

print("\nAccuracy Score on training data :" , training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction , Y_test)

print("\nAccuracy Score on testing data :" , testing_data_accuracy)

#Making a predictive system

input_data = (0.0491,0.0279,0.0592,0.1270,0.1772,
              0.1908,0.2217,0.0768,0.1246,0.2028,
              0.0947,0.2497,0.2209,0.3195,0.3340,
              0.3323,0.2780,0.2975,0.2948,0.1729,
              0.3264,0.3834,0.3523,0.5410,0.5228,
              0.4475,0.5340,0.5323,0.3907,0.3456,
              0.4091,0.4639,0.5580,0.5727,0.6355,
              0.7563,0.6903,0.6176,0.5379,0.5622,
              0.6508,0.4797,0.3736,0.2804,0.1982,
              0.2438,0.1789,0.1706,0.0762,0.0238,
              0.0268,0.0081,0.0129,0.0161,0.0063,
              0.0119,0.0194,0.0140,0.0332,0.0439)

# Changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Scale the input data using the same scaler
input_data_reshaped = scaler.transform(input_data_reshaped)

# Make prediction
prediction = model.predict(input_data_reshaped)
print("\nPredicted Label:", prediction[0])

# Interpreting result
if prediction[0] == 'R':
    print("The object is a Rock.")
else:
    print("The object is a Mine.")