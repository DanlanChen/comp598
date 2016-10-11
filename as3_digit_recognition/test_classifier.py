from sklearn.cross_validation import KFold
from sklearn import svm
from plots_as3 import plot_svm_accuracy, cfm_svm,plot_logistic_regression_accuracy,cfm_logistic_regression,cfm_svm_pca,plot_logistic_regression_accuracy_pca,cfm_logistic_regression_pca
from writeToKaggle import write_tokaggle
from calc_accuracy import accuracy
from sklearn import metrics,linear_model
import numpy as np
from load_numpy import getNumpy
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from preprocesses import pca
def svm1(X, X_valid, y, Y_valid,train_x,train_y,kaggle_test):
	#X,y-training set
	#X_valid -validationset
	#train_x,whole training set
	#X_test kaggle
	m, n = X.shape
	kf = KFold(m, n_folds=2)
	gammas = [0.001,0.01,0.1]
	cs = [1.0,10.0,100.0]
	accuracy_list = []
	for g in gammas:
		for c in cs:
			accs = []
			counter = 0 
			for train_index, test_index in kf:
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
				clf = svm.SVC(C= c,gamma=g)
				clf.fit(X_train,y_train)
				pred = clf.predict(X_test)
				# print type(pred)
				corr, acc = accuracy(y_test,pred)
				accs.append(acc)
				counter +=1
			acc_mean = np.mean(accs)
			accuracy_list.append((c,g,acc_mean))

	sortedaccuracy_list = sorted(accuracy_list, key = lambda x:x[2], reverse = True)
	print sortedaccuracy_list
	c_optimal = sortedaccuracy_list[0][0]
	g_optimal = sortedaccuracy_list[0][1]
	fname = 'results/svm/svm_accuracy.png'
	plot_svm_accuracy(accuracy_list,fname)

	# check validationset
	print "testing on validationset"
	clf = svm.SVC(C= c_optimal,gamma=g_optimal)
	clf.fit(X,y)
	pred__valid_svm = clf.predict(X_valid)
	print "results of svm using "+str(c_optimal)+'c'+str(g_optimal)+'gamma' + ":\n%s\n" %(metrics.classification_report(Y_valid,pred__valid_svm))
	fname2 = 'results/svm/'+str(c_optimal)+'c'+str(g_optimal)+'gamma'+'cfm_svm.png'
	cfm_svm(pred__valid_svm,Y_valid,c_optimal,g_optimal,fname2)
	# classifcation of kaggle testset
	print "classifcation kaggle"
	clf = svm.SVC(C= c_optimal,gamma=g_optimal)
	clf.fit(train_x,train_y)
	pred_kaggle_svm = clf.predict(kaggle_test)
	fname3 = 'results/svm/svm_kaggle.csv'
	write_tokaggle(fname3,pred_kaggle_svm)
def logistic_regression(X, X_valid, y, Y_valid,train_x,train_y,kaggle_test):
	m, n = X.shape
	kf = KFold(m, n_folds=2)
	cs = [1.0,10.0,100.0]
	accuracy_list=[]
	for c in cs:
		accs = []
		counter = 0 
		for train_index, test_index in kf:
			print "counter" + str(counter)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			clf = linear_model.LogisticRegression(C = c)
			clf.fit(X_train,y_train)
			pred = clf.predict(X_test)
			corr, acc = accuracy(y_test,pred)
			accs.append(acc)
			counter +=1
		acc_mean = np.mean(accs)
		accuracy_list.append((c,acc_mean))
	sortedaccuracy_list = sorted(accuracy_list, key = lambda x:x[1], reverse = True)
	print sortedaccuracy_list
	c_optimal = sortedaccuracy_list[0][0]
	fname = 'results/logistic_regression/logistic_regression_accuracy.png'
	plot_logistic_regression_accuracy(accuracy_list,fname)
	# check validationset
	print "testing on validationset"
	clf = linear_model.LogisticRegression(C = c_optimal)
	clf.fit(X,y)
	pred_valid_logistic_regression = clf.predict(X_valid)
	print "results of LogisticRegression using "+str(c_optimal)+'Lambda regularization' + ":\n%s\n" %(metrics.classification_report(Y_valid,pred_valid_logistic_regression))
	fname2 = 'results/logistic_regression/'+str(c_optimal)+'lambda'+'cfm_logistic_regression.png'
	cfm_logistic_regression(pred_valid_logistic_regression,Y_valid,c_optimal,fname2)
	# classifcation of kaggle testset
	print "classifcation kaggle"
	clf = linear_model.LogisticRegression(C = c_optimal)
	clf.fit(train_x,train_y)
	pred_kaggle_logistic = clf.predict(kaggle_test)
	fname3 = 'results/logistic_regression/logistic_regression_kaggle.csv'
	write_tokaggle(fname3,pred_kaggle_logistic)
def svm2_pca(X, X_valid, y, Y_valid,train_x,train_y,kaggle_test):
	pcas = [100,500,1000]
	gammas = [0.001,0.01,0.1]
	cs = [1.0,10.0,100.0]
	accuracy_list = []
	for pca in pcas:
		print 'pca',pca
		accuracy_list_p = []
		pca = PCA(n_components = pca,whiten = True)
		train_x_p = pca.fit_transform(train_x)

		X_p = pca.transform(X)
		X_valid_p = pca.transform(X_valid)
		kaggle_test_p = pca.transform(kaggle_test)
		m, n = X_p.shape
		kf = KFold(m, n_folds=2)
		for g in gammas:
			for c in cs:
				accs = []
				counter = 0 
				for train_index, test_index in kf:
					X_train, X_test = X_p[train_index], X_p[test_index]
					y_train, y_test = y[train_index], y[test_index]
					clf = svm.SVC(C= c,gamma=g)
					clf.fit(X_train,y_train)
					pred = clf.predict(X_test)
					# print type(pred)
					corr, acc = accuracy(y_test,pred)
					accs.append(acc)
					counter +=1
				acc_mean = np.mean(accs)
				accuracy_list.append((pca,c,g,acc_mean))
				accuracy_list_p.append((c,g,acc_mean))
		fname = 'results/svm_pca/svm_pca_accuracy_'+str(pca)+'pca.png'
		plot_svm_accuracy(accuracy_list_p,fname)

	sortedaccuracy_list = sorted(accuracy_list, key = lambda x:x[3], reverse = True)
	print sortedaccuracy_list
	pca_optimal = sortedaccuracy_list[0][0]
	c_optimal = sortedaccuracy_list[0][1]
	g_optimal = sortedaccuracy_list[0][2]

	pca = PCA(n_components = pca_optimal,whiten = True)
	train_x_p = pca.fit_transform(train_x)
	X_p = pca.transform(X)
	X_valid_p = pca.transform(X_valid)
	kaggle_test_p = pca.transform(kaggle_test)
	# check validationset
	print "testing on validationset"

	clf = svm.SVC(C= c_optimal,gamma=g_optimal)
	clf.fit(X_p,y)
	pred__valid_svm = clf.predict(X_valid_p)
	print "results of svm using pca "+str(pca_optimal)+'pca components'+str(c_optimal)+'c'+str(g_optimal)+'gamma' + ":\n%s\n" %(metrics.classification_report(Y_valid,pred__valid_svm))
	fname2 = 'results/svm_pca/'+str(pca_optimal)+'pca components'+str(c_optimal)+'c'+str(g_optimal)+'gamma'+'cfm_svm_pca.png'
	cfm_svm(pred__valid_svm,Y_valid,c_optimal,g_optimal,fname2)
	# classifcation of kaggle testset
	print "classifcation kaggle"
	clf = svm.SVC(C= c_optimal,gamma=g_optimal)
	clf.fit(train_x_p,train_y)
	pred_kaggle_svm = clf.predict(kaggle_test_p)
	fname3 = 'results/svm_pca/svm_pca_kaggle.csv'
	write_tokaggle(fname3,pred_kaggle_svm)
def svm_3_pca(train_x,train_y,kaggle_test):
	pcas = [100,500,1000]
	gammas = [0.001,0.01,0.1]
	cs = [1.0,10.0,100.0]
	accuracy_list = []
	m,n = train_x.shape
	kf = KFold(m, n_folds=2)
	for k in pcas:
		print k,'pca'
		accuracy_list_p = []
		train_x_transform,pca_f = pca(train_x,k)
		kaggle_test_transform = pca_f.transform(kaggle_test)
		# X_train, X_valid, Y_train, Y_valid = train_test_split(train_x_transform, train_y,test_size=0.2,random_state=0)
		for g in gammas:
			for c in cs:
				accs = []
				counter = 0 
				for train_index, test_index in kf:
					X_train, X_test,y_train, y_test = train_x[train_index],train_x[test_index],train_y[train_index],train_y[test_index]
					clf = svm.SVC(C= c,gamma=g)
					clf.fit(X_train,y_train)
					pred = clf.predict(X_test)
					# print type(pred)
					corr, acc = accuracy(y_test,pred)
					accs.append(acc)
					print "results of svm using pca "+str(counter)+'counter '+str(k)+'pca components '+str(c)+'c'+str(g)+'gamma ' + ":\n%s\n" %(metrics.classification_report(y_test,pred))
					fname2 = 'results/svm_pca/'+str(counter)+'counter'+str(k)+'pca components'+str(c)+'c'+str(g)+'gamma'+'cfm_svm_pca.png'
					cfm_svm_pca(pred,y_test,c,g,k,fname2)
					counter +=1

				acc_mean = np.mean(accs)
				accuracy_list.append((k,c,g,acc_mean))
				accuracy_list_p.append((c,g,acc_mean))

		fname = 'results/svm_pca/svm_pca_accuracy_'+str(k)+'pca.png'
		plot_svm_accuracy(accuracy_list_p,fname)

	sortedaccuracy_list = sorted(accuracy_list, key = lambda x:x[3], reverse = True)
	print sortedaccuracy_list
	pca_optimal = sortedaccuracy_list[0][0]
	c_optimal = sortedaccuracy_list[0][1]
	g_optimal = sortedaccuracy_list[0][2]

	train_x_transform,pca_f = pca(train_x,pca_optimal)
	kaggle_test_transform = pca_f.transform(kaggle_test)

	print "classifcation kaggle"
	clf = svm.SVC(C= c_optimal,gamma=g_optimal)
	clf.fit(train_x_transform,train_y)
	pred_kaggle_svm = clf.predict(kaggle_test_transform)
	fname3 = 'results/svm_pca/svm_pca_kaggle.csv'
	write_tokaggle(fname3,pred_kaggle_svm)
def logistic_regression2_pca(train_x,train_y,kaggle_test):
	pcas = [100,500,1000]
	m,n = train_x.shape
	kf = KFold(m, n_folds=2)
	cs = [1.0,10.0,100.0]
	accuracy_list=[]

	for k in pcas:
		for c in cs:
			accs = []
			counter = 0 
			for train_index, test_index in kf:
				print "counter" + str(counter)
				X_train, X_test,y_train, y_test = train_x[train_index],train_x[test_index],train_y[train_index],train_y[test_index]
				clf = linear_model.LogisticRegression(C = c)
				clf.fit(X_train,y_train)
				pred = clf.predict(X_test)
				corr, acc = accuracy(y_test,pred)
				accs.append(acc)
				print "results of logistic_regression using pca "+str(counter)+'counter '+str(k)+'pca components '+str(c)+'lambda'+":\n%s\n" %(metrics.classification_report(y_test,pred))
				fname2 = 'results/logistic_regression_pca/'+str(counter)+'counter'+str(k)+'pca components'+str(c)+'lambda'+'cfm_logistic_regression_pca.png'
				cfm_logistic_regression_pca(pred,y_test,c,k,fname2)
				counter += 1

			acc_mean = np.mean(accs)
			accuracy_list.append((k,c,acc_mean))
	sortedaccuracy_list = sorted(accuracy_list, key = lambda x:x[2], reverse = True)
	print sortedaccuracy_list
	c_optimal = sortedaccuracy_list[0][1]
	pca_optimal = sortedaccuracy_list[0][0]
	fname = 'results/logistic_regression_pca/logistic_regression_accuracy_pca.png'
	plot_logistic_regression_accuracy_pca(accuracy_list,fname)

	train_x_transform,pca_f = pca(train_x,pca_optimal)
	kaggle_test_transform = pca_f.transform(kaggle_test)

	print "classifcation kaggle"
	clf = linear_model.LogisticRegression(C= c_optimal)
	clf.fit(train_x_transform,train_y)
	pred_kaggle_svm = clf.predict(kaggle_test_transform)
	fname3 = 'results/logistic_regression_pca/logistic_regression_pca_kaggle.csv'
	write_tokaggle(fname3,pred_kaggle_svm)

# def main():
# 	train_x,train_y,kaggle_test = getNumpy()
# 	train_x = train_x[:100]
# 	train_y = train_y[:100]
# 	kaggle_test = kaggle_test[:100]
# 	logistic_regression2_pca(train_x, train_y, kaggle_test)

# def main():
# 	train_x,train_y,test_x = getNumpy()
# 	train_x = train_x[:10]
# 	train_y = train_y[:10]
# 	test_x = test_x[:10]
# 	X_train, X_valid, Y_train, Y_valid = train_test_split(train_x, train_y,test_size=0.2,random_state=0)
# 	print 'logistic_regression'
# 	logistic_regression(X_train,X_valid,Y_train,Y_valid,train_x,train_y,test_x)
# if __name__ == '__main__':
# 	main()







