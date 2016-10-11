from load_numpy import getNumpy
from test_classifier import svm1,logistic_regression
from sklearn.cross_validation import train_test_split
train_x,train_y,test_x = getNumpy()

# train_x = train_x[:10]
# train_y = train_y[:10]
# test_x = test_x[:10]
X_train, X_valid, Y_train, Y_valid = train_test_split(train_x, train_y,test_size=0.2,random_state=0)

print 'svm'
svm1(X_train,X_valid,Y_train,Y_valid,train_x,train_y,test_x)