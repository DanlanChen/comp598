
import sys
import cPickle
import gzip
from sklearn.cross_validation import train_test_split
import csv
import numpy as np
if __name__ == '__main__':
	train_inputs = []
	with open('/Users/chendanlan/Documents/comp598/as3/data_and_scripts/train_inputs.csv', 'rb') as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    next(reader, None)  # skip the header
	    for train_input in reader: 
	        train_input_no_id = []
	        for dimension in train_input[1:]:
	            train_input_no_id.append(float(dimension))
	        train_inputs.append(np.asarray(train_input_no_id))
	 # Load all training ouputs to a python list
	train_outputs = []
	with open('/Users/chendanlan/Documents/comp598/as3/data_and_scripts/train_outputs.csv', 'rb') as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    next(reader, None)  # skip the header
	    for train_output in reader:  
	        train_output_no_id = int(train_output[1])
	        train_outputs.append(train_output_no_id)

	 #Load all training inputs to a python list
	test_inputs = []
	with open('/Users/chendanlan/Documents/comp598/as3/data_and_scripts/test_inputs.csv', 'rb') as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    next(reader, None)  # skip the header
	    for test_input in reader: 
	        test_input_no_id = []
	        for dimension in test_input[1:]:
	            test_input_no_id.append(float(dimension))
	        test_inputs.append(np.asarray(test_input_no_id)) 
	# m =len(test_inputs)
	test_inputs = np.asarray(test_inputs,dtype = 'f')
	# test_output_initial = np.asarray([0]*m)
	# test_kaggle_set = (test_inputs,test_output_initial)
	# print test_kaggle_set
	# print test_kaggle_set[0].shape
	# print test_kaggle_set[1].shape
	X_train, X_test, Y_train, Y_test = train_test_split(train_inputs, train_outputs,test_size=0.2,random_state=0)
	# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,test_size=0.2,random_state=0)
	X_train = np.asarray(X_train,dtype = 'f')
	Y_train = np.asarray(Y_train)
	# X_valid = np.asarray(X_valid,dtype = 'f')
	# Y_valid = np.asarray(Y_valid)
	X_test = np.asarray(X_test,dtype = 'f')
	Y_test= np.asarray(Y_test)
	train_set = (X_train,Y_train)
	# valid_set = (X_valid,Y_valid)
	test_set = (X_test,Y_test)
	# print train_set
	cPickle.dump(train_set, gzip.open('train.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
	# cPickle.dump(valid_set, gzip.open('valid.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(test_set, gzip.open('test.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(test_inputs, gzip.open('test_kaggle.pickle.gz','wb'), cPickle.HIGHEST_PROTOCOL)

