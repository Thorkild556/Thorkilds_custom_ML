# a naive bayes that takes both continuous data and classes

import pandas as pd
import numpy as np
import math 
from scipy import stats 

dataframe = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(dataframe, delimiter=",")

#print(df.head(5))

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



    return predictor_dfs, target_lists, target_labels

#print(split_data(X, Y))



def train_model(predictor_df, target):

    predictor_class_labels = []
    for i in predictor_df:
        if X.dtypes[i] == 'object':
            predictor_class_labels.append(labels(predictor_df[i]))
    

    split = split_data(predictor_df, target)
    predictor_dfs = split[0]
    target_labels = split[2]
    #first calculate priors
    prior_list = []
    length_data = 0
    for i in predictor_dfs:
        length_data += len(i)

    for i in predictor_dfs: # calculate priors:
        prior_list.append(len(i)/length_data)

    conditional_probs = []
    # then calculate conditional probabilities or likelyhood depending on whether it is a class-predictor or gaussian predictor.
    for index, i in enumerate(predictor_dfs):
        conditional_probs_df = []
        obj_index_counter = -1
        for j in i:
            if predictor_df.dtypes[j] == 'object': # if discrete variable /class variable make regular naive bayes
                obj_index_counter += 1
                # Calculating the probabilities for each predictor class
                label_percentage = []
                for l in predictor_class_labels[obj_index_counter]: # for every type of label make a label-list
                    counter = 1 #starting counter at 1 as you cannot do log(0)
                    #print(l)
                    for k in range(len(i[j])): # for every value in the current column
                        #print(len(i[j]))
                        if i[j][k] == l:
                            counter += 1
                    label = l
                    label_and_perc = {label: counter/len(i[j])}
                    label_percentage.append(label_and_perc)
                cur_class = j
                cond_prob_and_class = {cur_class: label_percentage, "dtype": "object"}
                conditional_probs_df.append(cond_prob_and_class)
            elif predictor_df.dtypes[j] == 'int64' or predictor_df.dtypes[j] == 'float64': #if float or int then use gaussian (i could have used bernouli for integers but it could also be stuff like a uniform distribution)
                cur_class = j
                mu_and_std = stats.norm.fit(i[j])
                conditional_prob_and_class = {cur_class: mu_and_std, "dtype": "int64"}
                conditional_probs_df.append(conditional_prob_and_class)
                
            # elif predictor_df.dtypes[j] == 'int64': #if int then use bernouili
            #     cur_class = j
            #     mu_and_std = stats.be.fit(i[j])
            #     conditional_prob_and_class = {cur_class: mu_and_std, "dtype": "float64"}
            #     conditional_probs_df.append(conditional_prob_and_class)

            cond_probs_df = {"drug": target_labels[index], "probs": conditional_probs_df}
        conditional_probs.append(cond_probs_df)
    return conditional_probs, prior_list



model1 = train_model(X_trainset, y_trainset)

print(model1)

#brainstorm:
# how do i gather the right conditionals?
# if i use a for loop through trained model the loop will go through the list of lists with conditional probabilities for each label. # aka the drugs
# the nested loop will go through the lists with conditional probabilities for each label. # aka the classes
#the next nested loop will go through the conditional probabilities for each label. 
#so i need to associate the predictor values of the row i am trying to predict with the right indeces of the trained model.
# i could do this by going through the same process as i did when i created the probabilities, in order to make the right index.x
# i could also develop my model to make a dictionary instead, this way it will be easier to read the model. 

#i developed my model to create dictionaries
#so now i just need a loop that goes through all the drugs and calculates the bayes-score from the conditional probabilities given the class labels that we see, also including adding in the prior prob.


print(X_testset.columns)

def predict(model, X_test_data):
    predictions = []
    for i in range(len(X_test_data)): #for each row in the test data
        bayes_score_list = []
        for index, drug in enumerate(model[0]): #calculate bayes_score for each drug, so we can later pick the biggest value out of the bayes scores.
            bayes_score = math.log(model[1][index])
            #drug.get('drug'), bayes_score
            for prob in drug.get('probs'):
                if prob.get('dtype') == 'int64':
                    print("hej")
                    
                elif prob.get('dtype') == 'object':
                    continue
            return

            

        #return

                #elif X_testset.dtypes[j] == 'int64' or X_testset.dtypes[j] == 'float64':




        return predictions
        

    

    
                    


            #elif X.dtypes[j] == 'int64' or X.dtypes[j] == 'float64': # if continuous or integer make gausian naive bayes

predict(model1, X_testset)


