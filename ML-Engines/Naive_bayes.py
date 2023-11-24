# a naive bayes that takes both continuous data and classes

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


# print(X[0:5])
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
X_trainset = X_trainset.reset_index(drop=True) # resetting the indexes
y_trainset = y_trainset.reset_index(drop=True) # resetting the indexes

X_testset = X_testset.reset_index(drop=True) # resetting the indexes
y_testset = y_testset.reset_index(drop=True) # resetting the indexes

# get the labels of a predictor/class
def labels(Class):
    class_labels = []
    for i in Class:
        if i not in class_labels:
            class_labels.append(i)
    return class_labels

# print(labels(X['BP']))
    
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
    for i in range(len(predictor_dfs)):
        predictor_dfs[i] = predictor_dfs[i].reset_index(drop=True) # resetting the indexes



    return predictor_dfs, target_lists

#print(split_data(X, Y))



def train_model(predictor_df, target):

    predictor_class_labels = []
    for i in predictor_df:
        if X.dtypes[i] == 'object':
            predictor_class_labels.append(labels(predictor_df[i]))
    


    predictor_dfs = split_data(predictor_df, target)[0]
    #first calculate priors
    prior_list = []
    length_data = 0
    for i in predictor_dfs:
        length_data += len(i)

    for i in predictor_dfs: # calculate priors:
        prior_list.append(len(i)/length_data)

    conditional_probs = []
    # then calculate conditional probabilities or likelyhood depending on whether it is a class-predictor or gaussian predictor.
    for i in predictor_dfs:
        conditional_probs_df = []
        index_counter = -1
        for j in i:
            if X.dtypes[j] == 'object': # if discrete variable /class variable make regular naive bayes
                index_counter += 1
                # Calculating the probabilities for each predictor class
                label_percentage = []
                for l in predictor_class_labels[index_counter]: # for every type of label make a label-list
                    counter = 0 #
                    #print(l)
                    for k in range(len(i[j])): # for every value in the current column
                        #print(len(i[j]))
                        if i[j][k] == l:
                            counter += 1
                    label_percentage.append(counter/len(i[j]))
                conditional_probs_df.append(label_percentage)
            else:
                continue
        conditional_probs.append(conditional_probs_df)
    return conditional_probs, prior_list



model1 = train_model(X_trainset, y_trainset)

print(model1)

def predict(model, test_data):
    pass
                    


            #elif X.dtypes[j] == 'int64' or X.dtypes[j] == 'float64': # if continuous or integer make gausian naive bayes




