import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import neural_network

startTime = time.time()

labeled_images = pd.read_csv('mnist_train.csv')
images = labeled_images.iloc[0:10000,1:]
labels = labeled_images.iloc[0:10000,:1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
train_images /= 255.0
test_images /= 255.0

clf = neural_network.MLPClassifier()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images, test_labels))
print("---%s segundos---" %(time.time()-startTime))