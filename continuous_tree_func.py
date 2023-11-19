#https://www.youtube.com/watch?v=5O8HvA9pMew

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

# for i in X:
#     print(i)

#her transformere jeg dataen af de værdier der kun viser strings til dummy variables
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X['Sex'] = le_sex.transform(X['Sex']) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X['BP'] = le_BP.transform(X['BP'])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X['Cholesterol'] = le_Chol.transform(X['Cholesterol']) 

print(X[0:5])




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


print(best_split(X, Y))