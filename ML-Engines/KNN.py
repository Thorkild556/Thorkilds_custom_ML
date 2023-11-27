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






def predict(X_train, Y_train, X_test, k = 5):
    norm_list = []
    for column in X_train.columns:
        mu_and_std = stats.norm.fit(X_train[column])
        norm_list.append(mu_and_std)


    predictions = []
    for row in range(len(X_test)): # for each row in the test data
        dist_list = []
        counter = 0 #counter to keep track of which predictor
        z_scores = [] # the z_scores for each variable_value in the current row
        for val in X_test.loc[row,]: #for each value in the row
            Z_score_val = (val-norm_list[counter][0])/norm_list[counter][1]
            z_scores.append(Z_score_val)
            counter += 1
            # print(Z_score_val)
        for row_train in range(len(X_train)): #for each row in the train data: get the distance from the current row
            counter2 = 0 #counter to keep track of which predictor again
            z_scores2 = []
            for xval in X_train.loc[row_train,]: #get z scores for each variable
                xZ_score_val = (xval-norm_list[counter2][0])/norm_list[counter2][1]
                z_scores2.append(xZ_score_val)
                counter2 += 1
            distance_squared = 0
            for i in range(len(z_scores)): # calculate euclidian distance 
                distance_squared += (z_scores[i]-z_scores2[i])**2 #the distastanc
            distance = math.sqrt(distance_squared)
            dist_list.append(distance)

        #get the k nearest neigbors
        k_nearest = Y_train[np.argsort(dist_list)[0:k]]

        labels = {}
        for i in k_nearest: # count each label
            if i in labels: 
                labels[i] +=1
            else:
                labels[i] = 1
        
        prediction = list(labels.keys())[np.argmax(list(labels.values()))]
        predictions.append(prediction)

            
    return predictions




predicted_set = predict(X_trainset, y_trainset, X_testset, k = 6)



def accuracy(predicted, y_test):
    true_counter = 0
    for i, j in zip(predicted, y_test):
        if i == j:
            true_counter += 1
    accuracy_percent = true_counter/len(y_test)

    return accuracy_percent




print("\n", "KNN:", accuracy(predicted_set, y_testset))

from sklearn.metrics import confusion_matrix

print("Confusion Matrix: \n",confusion_matrix(predicted_set, y_testset))


