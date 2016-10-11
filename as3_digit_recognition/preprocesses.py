from sklearn.decomposition import PCA
def pca(X,k):
	pca = PCA(n_components = k,whiten = True)
	X_transform = pca.fit_transform(X)
	return X_transform,pca