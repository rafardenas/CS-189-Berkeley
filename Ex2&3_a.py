#Initial confirmation that all is gonna work

"""import sys
if sys.version_info[0] < 3:

	raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
from scipy import io

for data_name in ["mnist", "spam", "cifar10"]:
	data = io.loadmat("data/%s_data.mat" % data_name)
	print(f"\nloaded {data_name} data!")
	fields = "test_data", "training_data", "training_labels"
	for field in fields:
		print(field, data[field].shape)"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
from scipy import io
import seaborn as sns

#Ex2 -- 
#a) Write code that sets aside 10,000 training images as a validation set.

"""

print(type(mnist_data))
a = mnist_data.keys()
print(a)

training = mnist_data['training_data']
print(type(training))

print(len(training[0]))
""" 
mnist_data = io.loadmat("data/mnist_data.mat")
def train_test_split_mnist(data, train_size):
	#function to split training mnist data into training and test set 
	#training images are taken random
	x_train = []
	y_train = []
	features = data['training_data']
	labels = data['training_labels']
	rndm = np.random.choice(range(len(features)), train_size, replace=False)
	for i in rndm:
		feat = features[i]
		lab = int(labels[i])
		x_train.append(feat)
		y_train.append(lab)
	
	
	x_train = np.array(x_train, dtype = int)
	y_train = np.array(y_train, dtype = int)
	x_test = np.delete(features, rndm, axis = 0)
	y_test = np.delete(labels, rndm, axis = 0)
	return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split_mnist(mnist_data, 10000)



#b) For the spam dataset, write code that sets aside 20% of the training data as a validation set


spam_data = io.loadmat("data/spam_data.mat")
def train_test_split_spam(data, train_size):
	#function to split training mnist data into training and test set
	#input training_size in the range 0 to 1
	x_train = []
	y_train = []
	features = data['training_data']
	labels = data['training_labels']
	#ratio = round(len(features) * train_size)
	rndm = np.random.choice(range(len(features)), train_size, replace=False)
	for i in rndm:
		feat = features[i]
		lab = labels[i]
		x_train.append(feat)
		y_train.append(lab)

	x_train = np.array(x_train, dtype = int)
	y_train = np.array(y_train, dtype = int)
	x_test = np.delete(features, rndm, axis = 0)
	y_test = np.delete(labels, rndm, axis = 0)
	return x_train, x_test, y_train, y_test

	print(f'Size of validation set is {len(x_test)}')
x_train, x_test, y_train, y_test = train_test_split_spam(spam_data, 1)



#c) write code that sets aside 5,000 training images as a validation set.


cifar10_data = io.loadmat("data/cifar10_data.mat")

def train_test_split_cifar10(data, train_size):
	#function to split training mnist data into training and test set 
	#training images are taken random
	x_test = []
	y_test = []
	features = data['training_data']
	labels = data['training_labels']
	rndm = np.random.choice(range(len(features)), train_size, replace=False)
	for i in rndm:
		feat = features[i]
		lab = labels[i]
		x_test.append(feat)
		y_test.append(lab)


	x_train = np.delete(features, rndm, axis = 0)
	y_train = np.delete(labels, rndm, axis = 0)
	return x_train, x_test, y_train, y_test

	print(f'Size of validation set is {len(x_test)}')

x_train, x_test, y_train, y_test = train_test_split_cifar10(cifar10_data, 5000)


#Ex3. SVM



from sklearn.svm import SVC

def SupportVM(data, train_size):
	#function that uses SVM out of the bo with 'linear' kernel
	#test data by default is the one contained on the mnist dataset dictionary
	x_train, x_test, y_train, y_test = train_test_split_mnist(data, train_size)
	clf = SVC(kernel = 'linear') #not hyp now
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	#print(y_test.shape)
	return y_pred, y_test

#to check the accuracy of the model



training_examples = [100, 200] #, 500, 1000, 2000, 5000, 10000]

def Ex3_a(training_examples):
	y_pred, y_test = SupportVM(mnist_data, training_examples)
	#getting the accuracy
	from sklearn.metrics import accuracy_score
	acc = accuracy_score(y_test, y_pred)
	return acc
'''

accuracy_mnist = []
for size in training_examples:
	accu = Ex3_a(size)
	accuracy_mnist.append(accu)
	print("Accuracy is {} for {} training_examples".format(accu, size))

ax = sns.lineplot(x = training_examples, y = accuracy_mnist)
plt.xlabel('# of Training Examples')
plt.ylabel('Accuracy')
plt.show()

'''






