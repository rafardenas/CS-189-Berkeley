from Ex2and3_a import train_test_split_spam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import io
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
plt.style.use('ggplot')

spam_data = io.loadmat("data/spam_data.mat")
training_data = spam_data['training_data']
training_labels = spam_data['training_labels']



def CrossValidation(data_x, data_y, k):
    disjoint_x = []
    disjoint_y = []

    #this is a function to separate the sets into the 5 pieces
    size_subset = round(len(data_x) / 5)
    trunk_x = data_x
    trunk_y = data_y

    for i in range(k):
        #for the features

        size_set = len(trunk_x)
        print(len(trunk_x))
        rndm = np.random.choice(range(size_set), round(size_subset), replace=False)
        branch = np.delete(trunk_x, rndm, axis = 0) #set with the deleted selected indexes by rndm, kind of big
        diff = np.setdiff1d(trunk_x, branch) #portion of the set that is separated
        disjoint_x.append(diff)
        trunk_x = branch
        #for the labels

        size_set = len(trunk_y)
        print(len(trunk_y))
        rndm = np.random.choice(range(size_set), round(size_subset), replace=False)
        branch = np.delete(trunk_y, rndm, axis = 0) #set with the deleted selected indexes by rndm, kind of big
        diff = np.setdiff1d(trunk_y, branch) #portion of the set that is separated
        disjoint_y.append(diff)
        trunk_y = branch

    return disjoint_x, disjoint_y

x_sets, y_sets = CrossValidation(training_data, training_labels, 5)
print(len(x_sets[4])) 
#>>> 5
print(x_sets)

"Now to train based on cross validation protocol"
'''
spam_data = io.loadmat("data/spam_data.mat")

def CrossTraining(x_sets, y_sets, reg_param):
    accu_scores = []
    order = np.random.choice(range(len(sets)), len(sets), replace=False)
    clf = SVC(C = reg_param, kernel = 'linear')
    for i in order:
        validate_x = x_sets[i]
        validate_y = y_sets[i].reshape(-1,1)
        for j in order:
            if i != j:
                train_x = x_sets[j]
                train_y = y_sets[j].reshape(-1,1)

                clf.fit(train_x, train_y)
        predict = clf.predict(validate_x)
        accuracy = accuracy_score(validate_y, predict)
        accu_scores.append(accuracy)
    return accuracy

accu = CrossTraining(x_sets, y_sets, 1)


'''


"""
a = [1,2,3,4,5,6,7,8,9]
b = np.delete(a, [0,1,2,3])
print(b)
c = a[1,2]
print(c)

"""

