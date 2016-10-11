from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from writeToKaggle import write_tokaggle
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import sys
import numpy as np
import gzip
from keras.datasets.data_utils import get_file
from six.moves import cPickle
import sys

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''
# def loadNumpy(file):
# 	x = np.load(file)
# 	return x


# def getNumpy():
# 	file1 = 'train_inputs.npy'
# 	file2 = 'train_outputs.npy'
# 	file3 = 'test_inputs.npy'
# 	train_x = loadNumpy(file1)
# 	train_y = loadNumpy(file2)
# 	test_x = loadNumpy(file3)
# 	print (train_x.shape)
# 	print (train_y.shape)
# 	print (test_x.shape)
# 	return train_x,train_y,test_x
def load_data(path):
    # path = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding="bytes")

    f.close()

    return data  # (X_train, y_train), (X_test, y_test)
X_train, y_train =load_data('train.pickle.gz')
X_test, y_test = load_data('test.pickle.gz')
kaggle_test_x = load_data('test_kaggle.pickle.gz')
# X_train,y_train,X_test = getNumpy()
# X_train = X_train[:100]
# y_train = y_train[:100]
# X_test = X_test[:100]
# y_test = y_test[:100]
# kaggle_test_x = kaggle_test_x[:100]
batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
print ('epoch',nb_epoch)
# the data, shuffled and split between tran and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
kaggle_test_x = kaggle_test_x.reshape(kaggle_test_x.shape[0],1,img_rows, img_cols)
kaggle_test_x = kaggle_test_x.astype("float32")
# X_train /= 255
# X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Convolution2D(2*nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Convolution2D(4*nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
pred = model.predict_classes(kaggle_test_x,verbose=0)
# print (pred)
fname = "cnn_keras_new"+ str(nb_epoch)+'epoch.csv'
write_tokaggle(fname,pred)
print('Test score:', score[0])
print('Test accuracy:', score[1])
