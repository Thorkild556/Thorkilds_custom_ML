import pandas as pd




import pandas as pd
import numpy as np
import math 

dataframe = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(dataframe, delimiter=",")

#print(df.head(5))

counter = 0

# for
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
Y = df["Drug"]


#her transformere jeg dataen af de værdier der kun viser strings til dummy variables
from sklearn import preprocessing #only imported for the preprocessing

le_sex = preprocessing.LabelEncoder() 
le_sex.fit(['F','M'])
X['Sex'] = le_sex.transform(X['Sex']) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X['BP'] = le_BP.transform(X['BP'])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X['Cholesterol'] = le_Chol.transform(X['Cholesterol']) 

# print(X[0:5])
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
X_trainset = X_trainset.reset_index(drop=True) # resetting the indexes
y_trainset = y_trainset.reset_index(drop=True) # resetting the indexes

X_testset = X_testset.reset_index(drop=True) # resetting the indexes
y_testset = y_testset.reset_index(drop=True) # resetting the indexes


def labels(Class):
    class_labels = []
    for i in Class:
        if i not in class_labels:
            class_labels.append(i)
    return class_labels
    
    
def split_data(predictor_df, target):
    target_lists = []
    predictor_dfs = []
    target_labels = labels(target)
    for i in range(len(target_labels)):
        target_lists.append([])
        predictor_dfs.append(pd.DataFrame())
    for i in range(len(target_labels)):
        for j in range(len(target)):
            if target_labels[i] == target[j]:
                target_lists[i].append(target[j])
                predictor_dfs[i] = pd.concat([predictor_dfs[i], predictor_df.loc[[j]]])

    return predictor_dfs, target_lists

print(split_data(X, Y))



def get_percentages(predictor_df, target):
    pass