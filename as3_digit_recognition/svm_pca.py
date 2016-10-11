from test_classifier import svm_3_pca
from load_numpy import getNumpy
from sklearn.cross_validation import train_test_split
def main():
	train_x,train_y,kaggle_test = getNumpy()
	print 'svm_3_pca'
	# train_x = train_x[:100]
	# train_y = train_y[:100]
	# kaggle_test = kaggle_test[:100]
	svm_3_pca(train_x, train_y, kaggle_test)
if __name__ == '__main__':
	main()