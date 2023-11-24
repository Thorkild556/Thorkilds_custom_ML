#https://www.youtube.com/watch?v=5O8HvA9pMew

import pandas as pd
import numpy as np
import math 

dataframe = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(dataframe, delimiter=",")

#print(df.head(5))



 
### data preprocessing:
#counter = 0

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

# print(X_testset.head(60))


# print(X.head(5))
# print(X_trainset.head(5))
# print(y_trainset.head(5))


### Functions 

# entropy function
# the more variying amount of all 5 classes_labels or drugs, the less the entropy
def Entropy(Class):
    class_labels = []
    for i in Class:
        if i not in class_labels:
            class_labels.append(i)
    amount_list = []
    for i in class_labels:
        amount = 0
        for a in Class:
            if i == a:
                amount += 1
        amount_list.append(amount) # here i have made a list of how many of each predicter labels there is
    proportion = []
    for i in amount_list: #simply making a proportion list to make it more managable 
        proportion.append(i/len(Class)) #(it is a bit more computationally expensive that way as we could do the whole calculation in the next step instead)
    entropy = 0
    for i in proportion:
        entropy -= i*math.log2(i)
    return entropy, class_labels, proportion # i add the two last variables as well just so i dont have to make the same code over and over again.


#print(Entropy(Y))


# calculate information_gain only works if both are sorted ofc.
def information_gain(predictor, target_class):
    if len(predictor) != len(target_class):
        print("target vector and predictor is not the same length")

    entropy = Entropy(target_class)[0]

    result = Entropy(predictor) # i am doing this so i only have to call the function once
    predictor_labels, proportion_predictor_label = result[1], result[2] # the label and proportion of the label in the predicter class which we will use for calculation

    
    information_gains = entropy
    for i in range(len(predictor_labels)):
        label_list = [] #here we make a list that holds all the target_classes within the current predicter class so we can later calculate entropy of the target class within this predicter category/class_label
        for dat in range(len(predictor)):
            if predictor[dat] == predictor_labels[i]:
                label_list.append(target_class[dat])
        label_entropy_ = Entropy(label_list)[0]
        information_gains -= proportion_predictor_label[i]*label_entropy_
    
    return information_gains



def best_split(predictor_df, target):
    best_predictors = []
    best_predicter_threshhold = []
    for i in predictor_df: #for each variable/coulumn
        thresholds_information_gain = [] #make a list where we store one informationgain for each threshold (200 here)

        for j in predictor_df[i]: #for each value (j) in the current column we use that value as the threshold
            
            value_list = [] # starting a list so we can add binarry values
            for l in range(len(predictor_df)): #for each value (l) in current column we sort after 
                if predictor_df[i][l] <= j:
                    value_list.append('less')
                elif predictor_df[i][l] > j:
                    value_list.append('more')
            inf = information_gain(value_list, target)
            thresholds_information_gain.append(inf)

        best_predictors.append(np.max(thresholds_information_gain))
        best_predicter_threshhold.append(predictor_df[i][np.argmax(thresholds_information_gain)])

    return best_predictors, best_predicter_threshhold


#print(best_split(X, Y))


def gini_impurity(target):
    proportion = Entropy(target)[2]
    gini_impurity = 1
    for i in proportion:
        gini_impurity -= i*i
    
    return gini_impurity

#print(gini_impurity(Y))


def decision_tree(predictor_df, target, decisions=None, impurity = 0.65):
    if decisions is None: #
        decisions = []


    if gini_impurity(target) > impurity:
        highest_infogain = best_split(predictor_df, target)
        highest_infogain_index = np.argmax(highest_infogain[0])
        threshold = np.max(highest_infogain[1][highest_infogain_index])
        # return highest_infogain_thresh
        pred_name = Entropy(predictor_df)[1][highest_infogain_index] #the name of the predictor with the highest information_gain
        list_of_p_dfs = [pd.DataFrame(), pd.DataFrame()] #starting a list with the dataframes split
        list_of_targets = [pd.DataFrame(),pd.DataFrame()]

        decisions.append({"feature": pred_name, "Threshold <=": threshold, "action": "split"})

        for i in range(len(target)): #create a loop to split the target variable by the best predictor
            if predictor_df[pred_name][i] <= threshold:
                list_of_p_dfs[0] = pd.concat([list_of_p_dfs[0], predictor_df.loc[[i]]], ignore_index=True)
                list_of_targets[0] = pd.concat([list_of_targets[0], target.loc[[i]]], ignore_index=True)
            elif predictor_df[pred_name][i] > threshold:
                list_of_p_dfs[1] = pd.concat([list_of_p_dfs[1], predictor_df.loc[[i]]], ignore_index=True)
                list_of_targets[1] = pd.concat([list_of_targets[1], target.loc[[i]]], ignore_index=True)


        list_of_targets[0] = list_of_targets[0][0]
        list_of_targets[1] = list_of_targets[1][0]

        predictor_df = list_of_p_dfs #update the dataframe
        target = list_of_targets # update the target

        for i, (sub_df, sub_target) in enumerate(zip(list_of_p_dfs, list_of_targets)):
            sub_df, sub_target, decisions = decision_tree(sub_df, sub_target, impurity=impurity, decisions = decisions)
            list_of_p_dfs[i] = sub_df
            list_of_targets[i] = sub_target
    else:
        decisions.append({"feature": None, "action": "predict", "value": target.value_counts().idxmax()})
    
        #now recursion
        # for i, (sub_df, sub_target) in enumerate(zip(list_of_p_dfs, list_of_targets)): # for each (i is the index) sub_df and target_df in the dataframe- and target-lists :
        #     updated_dataframes, updated_targets = decision_tree(sub_df, sub_target, impurity=impurity) # recursion
        #     list_of_p_dfs[i] = updated_dataframes
        #     list_of_targets[i] = updated_targets

    return predictor_df, target, decisions

# X_trainset, X_testset, y_trainset, y_testset


#print(decision_tree(X_trainset, y_trainset, impurity = 0.20)[2])

#brainstorm to figure out how to predict:
#less than- or equal to threshhold is left and more is right so it runs the dataframe that is left firstly, if it cant anymore it goes one back up and to the right
#therfore i can count the number of times it has gone down and back up once, and how many times it has switched between thees two to know which of the predictions to make
# if it has gone down three times to the 4'th node but it goes up, i know the next leaf is the one to the right down from the 3'rd node. if it goes up again
#lets say you should go to the right. then ignore every thing until the code has went back as many times as it went to the left.

#how does it know not to continue if the node at the far left is the one?
#solution: making a variable keeping track of whether we are going left or right

def predict(decisions, X_test):
    predictions = []
    # for every row in the test_set, put it through the decision tree until the value hits a leaf note.
    for i in range(len(X_test)):
        left_counter = 0
        right_counter = 0
        Left = None
        right = None
        for j in decisions:
            if j['action'] == 'split': #if we are splitting the node then evaluate whether value for the predictor is under or over threshold
                if left_counter-right_counter == 0 and right == True: # because we want to start over if we have gone to the right.
                        Left = None
                        right = None
                        left_counter = 0 
                        right_counter = 0
                
                left_counter += 1
                if right == True:
                    continue
                threshhold = j['Threshold <=']
                if X_test[j['feature']][i] <= threshhold:
                    Left = True # keeping in mind whether we are going left or right
                    right = False
                    left_counter = 0
                    right_counter = 0
                    continue
                elif X_test[j['feature']][i] > threshhold:
                    right = True
                    Left = False
                    continue
            elif j['action'] == 'predict' and left_counter+right_counter != 0 and left_counter-right_counter == 0 and right == True: #if we have gone left and right equal amount of times
                predictions.append(j['value'])
                break
            elif j['action'] == 'predict' and Left == True:
                predictions.append(j['value'])
                break
            elif j['action'] == 'predict':
                right_counter += 1 
    return predictions
    #return predictions


decish = decision_tree(X_trainset, y_trainset, impurity = 0.20)[2]

predicted1 = predict(decish, X_testset)

print(predicted1)
print(y_testset)

print (decish, X_testset.loc[18,])

def accuracy(predicted, test):
    true_counter = 0
    for i, j in zip(predicted, test):
        if i == j:
            true_counter += 1

    accuracy_percent = true_counter/len(test)

    return accuracy_percent

print("\n", "DecisionTree's Accuracy", accuracy(predicted1, y_testset))

from sklearn.metrics import confusion_matrix

print("Confusion Matrix: \n",confusion_matrix(predicted1, y_testset))

# def evaluate



