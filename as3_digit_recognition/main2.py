from load_numpy import getNumpy
from test_classifier import svm1
from preprocesses import pca
from rbm import rbm
from sklearn.cross_validation import train_test_split
train_x,train_y,test_x = getNumpy()
train_x_transform = pca(train_x,100)
X_train, X_valid, Y_train, Y_valid = train_test_split(train_x_transform, train_y,test_size=0.2,random_state=0)

# svm1(train_x_transform,train_y)
rbm(train_x,train_y)