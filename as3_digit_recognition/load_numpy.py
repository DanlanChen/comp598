import numpy as np
def loadNumpy(file):
	x = np.load(file)
	return x
def getNumpy():
	file1 = 'train_inputs.npy'
	file2 = 'train_outputs.npy'
	file3 = 'test_inputs.npy'
	train_x = loadNumpy(file1)
	train_y = loadNumpy(file2)
	test_x = loadNumpy(file3)
	print train_x.shape
	print train_y.shape
	print test_x.shape
	return train_x,train_y,test_x

