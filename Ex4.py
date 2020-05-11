#Ex4

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from scipy import io
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')
np.random.seed(31415) #setting general random state

scaler = StandardScaler()
mnist_data = io.loadmat("data/mnist_data.mat")
from Ex2and3_a import train_test_split_mnist

#VEEEERY important the feature scaling, good explanation behind the reasoning:
# https://stackoverflow.com/questions/26225344/why-feature-scaling-in-svm
x_train, x_test, y_train, y_test = train_test_split_mnist(mnist_data, 10000) #10,000 is the training size
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Instructions: find the best C value

C_values = [1.e-5, 1.e-4, 1.e-2, 1, 1.e4, 1.e5, 1.e7]
accuracy = []



for i in C_values:
	print(i)
	clf = SVC(C = i, kernel = 'linear')
	clf = clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	accu = accuracy_score(y_test, y_pred)
	accuracy.append(accu)
	print(f'accuracy for with reg parameter {i} is {accu}')


#The accuracy does not change that much when C is large because it is almost the best decision boundary,
#C big means that the classifier will be "strict" ie will try to classify everythong almost perfect, making the boundaties
#closer to the main vector. But this would not generalize well depending on the case, here for example, we can see
#that an optimal accuracy is achieved with a relatively softer margin at C = 0.01.
#optimal meaning that wouldnt be overfitted (again, depends), and that it would perform more or less the same
#on a unseen data set






