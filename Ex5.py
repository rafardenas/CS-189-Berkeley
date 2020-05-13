from Ex2and3_a import train_test_split_spam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import io
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random

plt.style.use('ggplot')

spam_data = io.loadmat("data/spam_data.mat")
training_data = spam_data['training_data']
training_labels = spam_data['training_labels']
#print((training_labels == 0).sum())
#print((training_labels == 1).sum())


def CrossValidation(data_x, data_y):
    disjoint_x = []
    disjoint_y = []
    step = round(len(data_x) / 5)
    size_subset = len(data_x)
    random.seed()
    random.shuffle(data_x) 
    random.shuffle(data_y)
    
    x = 0
    for i in range (0, size_subset, step):
        x += 1
        j = i + step
        temp_x = data_x[i:j]
        temp_y = data_y[i:j]
        if x > 5:
            break
        disjoint_x.append(temp_x)
        disjoint_y.append(temp_y)

    return disjoint_x, disjoint_y


a, b = CrossValidation(training_data, training_labels)




#Now to train based on cross validation protocol"


def CrossTraining(x_sets, y_sets, reg_param):
    accu_scores = []
    order = np.random.choice(range(len(x_sets)), len(x_sets), replace=False)
    #order = [0,1,2,3,4]
    #print(len(order))
    
    clf = SVC(C = reg_param, kernel = 'linear')

    for i in order:
        
        validate_x = x_sets[i]

        validate_y = y_sets[i].reshape(-1,)

        for j in order:
            if i != j:
                train_x = x_sets[j]
                train_y = y_sets[j].reshape(-1,)
                try:
                    clf.fit(validate_x, validate_y)
                except:
                    pass
    predict = clf.predict(validate_x)
    accuracy = accuracy_score(validate_y, predict)
    accu_scores.append(accuracy)
    return accu_scores


#list at least 8 C values you tried, the corresponding accuracies, and the best C value.

C_values = [1.e-5, 1.e-4, 1.e-2, 1, 1.e2, 1.e4, 1.e5, 1.e7]
scores = []

for i in C_values:
    accu = CrossTraining(a, b, i)
    scores.append(accu)

cval_accu = zip(C_values, scores)
max_index = scores.index(max(scores))
max_cval = C_values[max_index]
#print(f"Maximum accuracy is {max(scores) with C value of {max_cval}})
#Final Result = max(scores)




