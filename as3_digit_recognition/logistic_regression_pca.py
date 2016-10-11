from load_numpy import getNumpy
from sklearn.cross_validation import train_test_split
from test_classifier import logistic_regression2_pca
def main():
	print 'logistic_regression2_pca'
	train_x,train_y,kaggle_test = getNumpy()
	# train_x = train_x[:100]
	# train_y = train_y[:100]
	# kaggle_test = kaggle_test[:100]
	logistic_regression2_pca(train_x, train_y, kaggle_test)
if __name__ == '__main__':
	main()