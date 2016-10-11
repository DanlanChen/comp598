from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model,metrics
from sklearn.pipeline import Pipeline
def rbm(X,Y):
	# Models we will use
	logistic = linear_model.LogisticRegression()
	rbm = BernoulliRBM(random_state=0, verbose=True)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=0)
	classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
	###############################################################################
	# Training

	# Hyper-parameters. These were set by cross-validation,
	# using a GridSearchCV. Here we are not performing cross-validation to
	# save time.
	rbm.learning_rate = 0.06
	rbm.n_iter = 1000
	# More components tend to give better prediction performance, but larger
	# fitting time
	rbm.n_components = 100
	logistic.C = 6000.0

	# Training RBM-Logistic Pipeline
	classifier.fit(X_train, Y_train)

	# Training Logistic regression
	logistic_classifier = linear_model.LogisticRegression(C=100.0)
	logistic_classifier.fit(X_train, Y_train)
	# Evaluation

	print()
	print("Logistic regression using RBM features:\n%s\n" % (
	    metrics.classification_report(
	        Y_test,
	        classifier.predict(X_test))))

	print("Logistic regression using raw pixel features:\n%s\n" % (
	    metrics.classification_report(
	        Y_test,
	        logistic_classifier.predict(X_test))))