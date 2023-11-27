# this is my knn-engine


import pandas as pd
import numpy as np
import math 
from scipy import stats 


###data and preproccessing

dataframe = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(dataframe, delimiter=",")

#print(df.head(5))

# for
X = df[['Age', 'Na_to_K']]
Y = df["Drug"]


# print(X[0:5])
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
X_trainset = X_trainset.reset_index(drop=True) # resetting the indexes
y_trainset = y_trainset.reset_index(drop=True) # resetting the indexes

X_testset = X_testset.reset_index(drop=True) # resetting the indexes
y_testset = y_testset.reset_index(drop=True) # resetting the indexes


### model training and prediction

def predict(X_train, Y_train, X_test):
    norm_list = []
    for column in X_train.columns:
        mu_and_std = stats.norm.fit(X_train[column])
        norm_list.append(mu_and_std)
    
    for val in range(len(X_test)): # for each row in the test data
        dist_list = []
        



predict(X_trainset, y_trainset, X_testset)