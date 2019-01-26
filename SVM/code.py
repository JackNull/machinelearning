import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cvxopt import matrix, solvers

from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

class HardMMSVM():
	def __init__(self):
		self.w = None
		self.b = 0

	def train(self, setx, sety):
		# train a classifier
		NUM = len(setx)
		P = matrix([[(np.dot(setx[i],setx[j]) * sety[i] * sety[j]) for i in range(NUM)] for j in range(NUM)])
		q = matrix(-np.ones((NUM, 1)))
		G = matrix(-np.eye(NUM))
		h = matrix(np.zeros(NUM))
		A = matrix(sety).trans()
		b = matrix(np.zeros(1))
		sol = solvers.qp(P, q, G, h, A, b)
		alphas = np.array(sol['x'])

		self.w = np.sum(alphas * sety[:, None] * setx, axis=0)
		self.b = - ( np.min(setx[sety==1] @ self.w.T) + np.max(setx[sety==-1] @ self.w.T) )/2

	def predict(self, setx):
		# predict using classifer w and b
		scores = setx @ self.w.T + self.b
		prediction = [1 if score > 0 else -1 for score in scores]
		return prediction

def generate_gaussian(m, c, num):
	return np.random.multivariate_normal(m, c, num)

def plot_data_with_labels(x, y, w, b):
	COLORS = ['red', 'blue']
	unique = np.unique(y)
	for li in range(len(unique)):
		x_sub = x[y == unique[li]]
		plt.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])

	slope = -w[0] / w[1]
	intercept = -b / w[1]
	x = np.arange(0, 6)
	plt.plot(x, x * slope + intercept, 'k-')
	plt.show()

def partbandc():
	dataset = pd.read_csv("creditCard.csv")
	features = list(dataset.columns)
	
	features.remove("Class")
	target = "Class"
	train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=2018)
	train_x, train_y = train_data[features], train_data[target]
	test_x, test_y = test_data[features], test_data[target]

	train_x = preprocessing.scale(train_x)
	test_x = preprocessing.scale(test_x)
	
	lsvm = svm.SVC(kernel="linear")
	lsvm.fit(train_x, train_y)
	accuracy = lsvm.score(test_x, test_y)
	scores = lsvm.decision_function(test_x)
	print("Accuracy is %.10f" % accuracy)

	fpr, tpr, threshold = metrics.roc_curve(test_y, scores)
	auc = metrics.roc_auc_score(test_y, scores)
	print("AUC is %.10f" % auc)
	plt.plot(fpr,tpr)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()

	#gsvm = svm.SVC(kernel="rbf", gamma=1/5)
	gsvm = svm.SVC(kernel="rbf", gamma=1/25)

	gsvm.fit(train_x, train_y)
	accuracy = gsvm.score(test_x, test_y)
	scores = gsvm.decision_function(test_x)
	print("Accuracy is %.10f" % accuracy)

	fpr, tpr, threshold = metrics.roc_curve(test_y, scores)
	auc = metrics.roc_auc_score(test_y, scores)
	print("AUC is %.10f" % auc)
	plt.plot(fpr,tpr)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.show()


if __name__ == "__main__":
	# part (a)

	# svm = HardMMSVM()

	# DIM = 2
	# NUM = 50
	# # 2-D mean of ones
	# M1 = np.ones((DIM,))
	# # 2-D mean of threes
	# M2 = 3 * np.ones((DIM,))
	# # 2-D covariance of 0.3
	# C1 = np.diag(0.3 * np.ones((DIM,)))
	# # 2-D covariance of 0.2
	# C2 = np.diag(0.2 * np.ones((DIM,)))
	# # generate 50 points from gaussian 1
	# x1 = generate_gaussian(M1, C1, NUM)
	# # labels
	# y1 = np.ones((x1.shape[0],))
	# # generate 50 points from gaussian 2
	# x2 = generate_gaussian(M2, C2, NUM)
	# y2 = -np.ones((x2.shape[0],))
	# # join
	# x = np.concatenate((x1, x2), axis = 0)
	# y = np.concatenate((y1, y2), axis = 0)
	# # print('x {} y {}'.format(x.shape, y.shape))
	# train_x = x
	# train_y = y
	# svm.train(train_x, train_y)
	# prediction = svm.predict(train_x)
	# print(prediction==train_y)

	# partbandc()





