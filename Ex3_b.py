#H1_Ex3 b).py

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from scipy import io
import seaborn as sns
plt.style.use('ggplot')
np.random.seed(31415) #setting general random state


from Hw1_2 import train_test_split_spam

spam_data = io.loadmat("data/spam_data.mat")
X = spam_data['training_data']
print(X.shape[0])
print(spam_data.keys())

#some exploring

#print(type(spam_data['training_labels']))
#print(spam_data['training_data'][0].shape)
from sklearn.metrics import accuracy_score
print(len(spam_data['test_data']))


training_size = [100, 200, 500, 1000, 2000, 4000]
accuracy = []
#training_size = [100]
for i in training_size:
	x_train, x_test, y_train, y_test = train_test_split_spam(spam_data, i)
	y_train = y_train.reshape(-1)
	y_test = y_test.reshape(-1)
	clf = SVC(kernel = 'linear')
	clf.fit(x_train, y_train)
	y_predict = clf.predict(x_test)
	accu = accuracy_score(y_test, y_predict)
	accuracy.append(accu)
	print("accuracy with {} samples is {}".format(i, accu))

plt.plot(training_size, accuracy, 'g-')
plt.xlabel("# Training examples")
plt.ylabel("Accuracy")
plt.show()

#print(y_test.shape)
#print(y_train.shape)





