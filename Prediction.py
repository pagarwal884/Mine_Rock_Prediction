#Importing the Dependencies

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