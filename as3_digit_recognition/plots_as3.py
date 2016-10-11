import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.mplot3d import Axes3D
def plot_svm_accuracy(accuracylist,fname):
	dir = os.path.dirname(fname)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	ax = Axes3D(fig)  
	x = []
	y = []
	z = []
	for t in accuracylist:
		x.append(t[0])
		y.append(t[1])
		z.append(t[2])
	ax.scatter(x, y, z,c = 'r', marker = 'o')
	ax.set_xlabel('penalty parameter c')
	ax.set_ylabel('gamma')
	ax.set_zlabel('accuracy')
	ax.set_title("svm")
	plt.savefig(fname,format = 'png')
	plt.close()
def cfm_svm(y_pred,y_test,c,g,fname2):
	dir = os.path.dirname(fname2)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cm = confusion_matrix(y_test, y_pred)
	cax = ax.matshow(cm)
	plt.title("Confusion matrix of svm  (C = %i,gamma = '%f')"
	          % (c, g))
	plt.colorbar(cax)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(fname2,format = 'png')
	plt.close()
def plot_logistic_regression_accuracy(accuracylist,fname):
	dir = os.path.dirname(fname)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	x = []
	y = []
	for t in accuracylist:
		x.append(t[0])
		y.append(t[1])
	plt.scatter(x,y)
	plt.title("logistic regression accuracy based on different regularization parameter")
	plt.ylabel('accuracy')
	plt.xlabel("lambda")
	plt.savefig(fname,format = 'png')
	plt.close()
def cfm_logistic_regression(y_pred,y_test,c,fname2):
	dir = os.path.dirname(fname2)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cm = confusion_matrix(y_test, y_pred)
	cax = ax.matshow(cm)
	plt.title("Confusion matrix of logistic_regression  (C = %i)"
	          % (c))
	plt.colorbar(cax)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(fname2,format = 'png')
	plt.close()
def cfm_logistic_regression_pca(y_pred,y_test,c,k,fname2):
	dir = os.path.dirname(fname2)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cm = confusion_matrix(y_test, y_pred)
	cax = ax.matshow(cm)
	plt.title("Confusion matrix of logistic_regression pca component is %i  (lambda = %f)"
	          % (k,c))
	plt.colorbar(cax)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(fname2,format = 'png')
	plt.close()
def cfm_svm_pca(y_pred,y_test,c,g,k,fname2):
	dir = os.path.dirname(fname2)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cm = confusion_matrix(y_test, y_pred)
	cax = ax.matshow(cm)
	plt.title("Confusion matrix of svm pca component is %i (C = %i,gamma = '%f')"
	          % (k,c, g))
	plt.colorbar(cax)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(fname2,format = 'png')
	plt.close()
def plot_logistic_regression_accuracy_pca(accuracylist,fname):
	dir = os.path.dirname(fname)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	fig = plt.figure()
	ax = Axes3D(fig)  
	x = []
	y = []
	z = []
	for t in accuracylist:
		x.append(t[0])
		y.append(t[1])
		z.append(t[2])
	ax.scatter(x, y, z,c = 'r', marker = 'o')
	ax.set_xlabel('PCA component')
	ax.set_ylabel('lambda')
	ax.set_zlabel('accuracy')
	ax.set_title("logistic regression accuracy based on different pca and lambda")
	plt.savefig(fname,format = 'png')
	plt.close()
