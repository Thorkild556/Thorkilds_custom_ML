# random forest with the decision tree algorithm already made
import pandas as pd
import numpy as np
import math 
from decision_tree_func_2_0 import decision_tree, predict


#https://www.youtube.com/watch?v=5O8HvA9pMew



dataframe = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(dataframe, delimiter=",")

#print(df.head(5))



 
### data preprocessing:
#counter = 0

# for
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
Y = df["Drug"]

X = X.copy() # so sklearn it doesnt print warning

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



decish = decision_tree(X_trainset, y_trainset, impurity = 0.20)[2]

predicted1 = predict(decish, X_testset)


def accuracy(predicted, test):
    true_counter = 0
    for i, j in zip(predicted, test):
        if i == j:
            true_counter += 1

    accuracy_percent = true_counter/len(test)

    return accuracy_percent

print("\n", "DecisionTree's Accuracy", accuracy(predicted1, y_testset))
