import numpy as np
import matplotlib.pyplot as plt

def question3b(X):
	beta1 = np.array([0.1, 0.3, 0.2, 0.2, 0.9, 0.8, 0.9, 0.1, 0.4, 0.2, 0.7, 0.3, 0.1, 0.7, 0.8, 0.3, 0.2, 0.8, 0.1, 0.7])
	beta2 = np.array([0.5, 0.6, 0.7, 0.9, 0.9, 0.8, 0.9, 0.8, 0.6, 0.5, 0.7, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.8, 0.5, 0.7])
	T = 100
	np.random.seed(2018)
	lambdas = np.arange(0, 5, 0.05)
	XTX = np.dot(np.transpose(X), X)
	
	MSE1 = np.zeros(len(lambdas))
	MSE2 = np.zeros(len(lambdas))
	for k, lamb in enumerate(lambdas):
		temp = np.dot(np.linalg.inv(XTX + lamb * np.identity(np.shape(X)[1])), np.transpose(X))
		error1 = np.zeros(T)
		error2 = np.zeros(T)
		for i in range(T):
			noise = np.random.normal(0,1,50)
			
			# generate response vector y
			Y1 = np.dot(X, beta1) + noise
			Y2 = np.dot(X, beta2) + noise
			# compute estimated beta
			estimator1 = np.dot(temp, Y1)
			estimator2 = np.dot(temp, Y2)

			# record the error
			error1[i] = np.linalg.norm(Y1 - np.dot(X, estimator1))**2
			error2[i] = np.linalg.norm(Y2 - np.dot(X, estimator2))**2

		MSE1[k] = np.sum(error1)/T
		MSE2[k] = np.sum(error2)/T

	plt.plot(lambdas, MSE1, label="beta1")
	plt.plot(lambdas, MSE2, label="beta2")
	plt.xlabel("lambda")
	plt.ylabel("MSE")
	plt.legend()
	plt.show()

def question3c(X):
	X = X / X.max(axis=0)
	Up, L, VT = np.linalg.svd(X, full_matrices=False)
	Li = L
	L = L * np.identity(len(L))
	beta1 = np.array([0.1, 0.3, 0.2, 0.2, 0.9, 0.8, 0.9, 0.1, 0.4, 0.2, 0.7, 0.3, 0.1, 0.7, 0.8, 0.3, 0.2, 0.8, 0.1, 0.7])
	T = 10
	np.random.seed(2018)
	gamma = np.dot(VT, beta1)
	alphaMatrix = 0.1 * np.ones(len(X))
	E_gamma = np.zeros(20)
	temp = np.dot(np.dot(np.linalg.inv(L**2 + np.identity(len(L))), L), Up.T)
	mse = np.zeros(20)
	for i in range(T):
		noise = np.random.normal(0,1,50)
		Y = alphaMatrix + np.dot(X, beta1) + noise

		estimator = np.dot(temp, Y)
		E_gamma = E_gamma + estimator
		mse = (estimator - gamma)**2 + mse
	
	E_gamma = E_gamma / T
	mse = mse / T
	temp = np.zeros(len(L))
	for i in range(len(L)):
		temp[i] = (Li[i]**2 + gamma[i]**2) / (Li[i]**2 + 1)**2

	xaxis = np.arange(1,21,1)
	plt.scatter(xaxis, gamma, label="gamma")
	plt.scatter(xaxis, E_gamma, label="E[gamma]")
	plt.xlabel("feature-p")
	plt.legend()
	plt.show()

	plt.scatter(xaxis, mse, label="MSE")
	plt.scatter(xaxis, temp, label="L")
	plt.xlabel("feature-p")
	plt.legend()
	plt.show()

	

if __name__ == '__main__':
	X = np.loadtxt("X.dat", delimiter=",")
	question3b(X)
	question3c(X)

