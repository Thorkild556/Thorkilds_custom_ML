import pandas as pd
import numpy as np
import math 

dataframe = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(dataframe, delimiter=",")

#print(df.head(5))

counter = 0

# for
X = df[['Sex', 'BP', 'Cholesterol']]
Y = df["Drug"]


# sub_df = pd.DataFrame()

 
# for dat in range(3):
#     sub_df = sub_df.append(X.loc[dat])

# print(sub_df.head(3))

# for i in X:
#     print(X[i])
        


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
        label_list = [] #here we make a list that holds all the tget_classes within the current predicter class so we can later calculate entropy of the target class within this predicter category/class_label
        for dat in range(len(predictor)):
            if predictor[dat] == predictor_labels[i]:
                label_list.append(target_class[dat])
        label_entropy_ = Entropy(label_list)[0]
        information_gains -= proportion_predictor_label[i]*label_entropy_
    
    return information_gains

        
        
#print(information_gain(X['BP'], Y))



def infgain_list(dataframe, target):
    information_gain_list = []
    for i in dataframe:
        inf = information_gain(dataframe[i], target)
        information_gain_list.append(inf)
    return information_gain_list


# #print(infgain_list(X, Y))
    

def gini_impurity(target):
    proportion = Entropy(target)[2]
    gini_impurity = 1
    for i in proportion:
        gini_impurity -= i*i
    
    return gini_impurity

#print(gini_impurity(Y))

counter = 0

# how would you calculate the bracnh you are currently in freom earlier itterations.


def decision_tree(dataframe, target, decisions=None, impurity = 0.63):
    if decisions is None:
        decisions = []

    if gini_impurity(target) > impurity:
        infgainlist = infgain_list(dataframe, target)
        big = dataframe.columns[np.argmax(infgainlist)] #, max(infgainlist)
        decisions.append({"feature": big, "threshold": None, "action": "split"})
        #create a new dataset for each class in the predictor with the biggest information-gain.
        predictor = dataframe[big] 
        predictor_labels = Entropy(predictor)[1]
        list_of_dataframes = []
        list_of_targets = []

        for i in predictor_labels:
            sub_df = pd.DataFrame()
            sub_target = pd.DataFrame()
            for dat in range(len(predictor)):
                if predictor[dat] == i:
                    sub_df = pd.concat([sub_df, dataframe.loc[[dat]]], ignore_index=True)
                    sub_target = pd.concat([sub_target, target.loc[[dat]]], ignore_index=True)
            sub_target = sub_target[0]
            list_of_dataframes.append(sub_df)
            list_of_targets.append(sub_target)

        dataframe = list_of_dataframes
        target = list_of_targets

        for i, (sub_df, sub_target) in enumerate(zip(list_of_dataframes, list_of_targets)):
            # Update dataframe and target only for the specific branch/sub-branch
            updated_dataframes, updated_targets, decisions = decision_tree(sub_df, sub_target, decisions, impurity=impurity)
            list_of_dataframes[i] = updated_dataframes
            list_of_targets[i] = updated_targets

    else:
        # If the node is a leaf node, record the decision
        decisions.append({"feature": None, "threshold": None, "action": "predict", "value": target.value_counts().idxmax()})

        # for i in range(len(predictor_labels)):
        #     sub_branch += 1
        #     lists[1] = sub_branch
        #     print("sub:", sub_branch)
        #     decision_tree(dataframe[i], target[i], branch, lists = lists)

    return dataframe, target, decisions

    
print(decision_tree(X,Y)[2])

# list_of_df, list_oftarget = div[0], div[1]

