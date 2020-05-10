#Ex4

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from scipy import io
import seaborn as sns
from sklearn.metrics import accuracy_score
plt.style.use('ggplot')
np.random.seed(31415) #setting general random state


mnist_data = io.loadmat("data/mnist_data.mat")
from Ex2and3_a import train_test_split_mnist

x_train, x_test, y_train, y_test = train_test_split_mnist(mnist_data, 100) #10,000 is the training size

#Instructions: find the best C value

C_values = [1, 25, 100]
accuracy = []




for i in C_values:
	print(i)
	clf = SVC(C = i, kernel = 'linear')
	clf = clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	accu = accuracy_score(y_test, y_pred)
	accuracy.append(accu)
	print(f'accuracy for with reg parameter {i} is {accu}')
			





	"""
clf = SVC(C = 40, kernel = 'linear')
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accu = accuracy_score(y_test, y_pred)
print(accu)

#7549415692821368
"""


