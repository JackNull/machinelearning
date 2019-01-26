# Toggle the commenting to test individual questions #

import numpy as np
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate a classifier
# @oaram runs number of epochs
# @param setx array of training data
# @param sety array of labels of training data in setx
# return a list of classifiers from epoch 1 to {runs}
def perceptron(runs, setx, sety):
	# initialize w and I
	w = np.zeros(setx.shape[1], dtype=np.float64)
	I = runs
	t = 0
	data_size = setx.shape[0]
	w_list = np.zeros((runs, setx.shape[1]))
	while t < I:
		for i in range(0, data_size):
			if np.dot(w, setx[i]) * sety[i] <= 0:
				w = np.add(w, sety[i] * setx[i])
		w_list[t] = w		
		t = t + 1
	return w_list

# perceptron for question 1d
def perceptron_1d(runs, setx, sety):
	# initialize w and I
	w = np.zeros(setx.shape[1], dtype=np.float64)
	I = runs
	t = 0
	done = False
	data_size = setx.shape[0]
	onethird = data_size / 3
	w_list = np.zeros((2, setx.shape[1]));
	while t < I:
		for i in range(0, data_size):
			if np.dot(w, setx[i]) * sety[i] <= 0:
				w = np.add(w, sety[i] * setx[i])
			if t == 0 and i > onethird and done == False:
				w_list[0] = w	
				done = True	
		t = t + 1
	w_list[1] = w
	return w_list

# predict the data using classifier w
# @param w classifier
# @param setx array of data for prediction
# return array of {-1,1}
def predict(w, setx):
	pred = np.zeros(setx.shape[0])
	data_size = setx.shape[0]
	for i in range(0, data_size):
		pred[i] = 1 if np.dot(w, setx[i]) > 0 else -1
	return pred

def predict_b(b, w, setx):
	pred = np.zeros(setx.shape[0])
	data_size = setx.shape[0]
	for i in range(0, data_size):
		pred[i] = 1 if np.dot(w, setx[i]) - b > 0 else -1
	return pred

# calculate the accurary based on prediction array and actual labels
# @param prediction array of {-1,1}
# @param array of actual labels
# return accuracy of the prediction
def accuracy(prediction, actual):
	TP = 0
	TN = 0
	data_size = prediction.size
	for i in range(0, data_size):
		if prediction[i] == 1 and actual[i] == 1:
			TP = TP + 1
		elif prediction[i] == -1 and actual[i] == -1:
			TN = TN + 1
	acc = (TP + TN) / data_size
	return acc

# calculate the True Positive Rate
# @param prediction array of {-1,1}
# @param array of actual labels
# return the True Positive Rate
def getTPR(prediction, actual):
	TP = 0
	FN = 0
	data_size = prediction.size
	for i in range(0, data_size):
		if prediction[i] == 1 and actual[i] == 1:
			TP = TP + 1
		elif prediction[i] == -1 and actual[i] == 1:
			FN = FN + 1
	TPR = TP / (TP + FN)
	return TPR

# calculate the False Positive Rate
# @param prediction array of {-1,1}
# @param array of actual labels
# return the False Positive Rate
def getFPR(prediction, actual):
	FP = 0
	TN = 0
	data_size = prediction.size
	for i in range(0, data_size):
		if prediction[i] == 1 and actual[i] == -1:
			FP = FP + 1
		elif prediction[i] == -1 and actual[i] == -1:
			TN = TN + 1
	FPR = FP / (FP + TN)
	return FPR

# compute the confusion matrix
# @param prediction array of {-1,1}
# @param array of actual labels
# return print out the confusion matrix in order of: TP, FP, FN, TN
def plotConfusionMatrix(prediction, actual):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	data_size = prediction.size
	for i in range(0, data_size):
		if prediction[i] == 1 and actual[i] == 1:
			TP = TP + 1
		elif prediction[i] == -1 and actual[i] == -1:
			TN = TN + 1
		elif prediction[i] == 1 and actual[i] == -1:
			FP = FP + 1
		elif prediction[i] == -1 and actual[i] == 1:
			FN = FN + 1
	print("TP = %d , FP = %d" % (TP, FP))
	print("FN = %d , TN = %d" % (FN, TN))

# balanced winnow algorithm to find a classifier
# @param eta tuning parameter
# @param runs maximum number of epochs
# @param setx array of training data
# @param sety array of labels of training data in setx
# return a list of classifiers from epoch 1 to {runs}
def balancedwinnow(eta, runs, setx, sety):
	p = setx.shape[1]
	w_p = np.empty(setx.shape[1])
	w_p.fill(1/(2*p))
	w_n = np.empty(setx.shape[1])
	w_n.fill(1/(2*p))
	I = runs
	t = 0
	data_size = setx.shape[0]
	w_list = np.zeros((runs, 2, setx.shape[1]))

	while t < I:
		for i in range(0, data_size):
			if sety[i] * (np.dot(w_p, setx[i]) - np.dot(w_n, setx[i])) <= 0:
				w_p = w_p * (math.e**(eta * sety[i] * setx[i]))
				w_n = w_n * (math.e**(-1 * eta * sety[i] * setx[i]))
				s = np.sum(w_n) + np.sum(w_p)
				w_p = w_p / s
				w_n = w_n / s
		w_list[t][0] = w_p
		w_list[t][1] = w_n
		t = t + 1
	return w_list

# predict the data using classifier w = (wp, wn)
# @param w classifier
# @param setx array of data for prediction
# return array of {-1,1}
def predict_winnow(w, setx):
	pred = np.zeros(setx.shape[0])
	data_size = setx.shape[0]
	for i in range(0, data_size):
		pred[i] = 1 if np.dot(w[0], setx[i]) - np.dot(w[1], setx[i]) > 0 else -1
	return pred



if __name__ == '__main__':

	mnist = input_data.read_data_sets("datasets/", one_hot = True)

	MAX_EPOCHS = 100
	#training data
	train_x = np.array(mnist.train.images)
	train_y = np.array(mnist.train.labels)
	indices = np.where(np.logical_or(train_y[:,4]==1, train_y[:,9]==1))

	train_x = train_x[indices]
	train_y = train_y[indices]
	train_y = np.array(list(map(lambda x : 2*x[4]-1, train_y)))
	
	#test data
	test_x = np.array(mnist.test.images)
	test_y = np.array(mnist.test.labels)
	test_indices = np.where(np.logical_or(test_y[:,4]==1, test_y[:,9]==1))
	test_x = test_x[test_indices]
	test_y = test_y[test_indices]
	test_y = np.array(list(map(lambda x : 2*x[4]-1, test_y)))

	#print(perceptron(1, train_x, train_y))
	#1(a)
	# t = 0
	# acc_list = np.zeros(MAX_EPOCHS)
	# epochs = np.zeros(MAX_EPOCHS)
	# w_list = perceptron(MAX_EPOCHS, train_x, train_y)
	# while t < MAX_EPOCHS:
	# 	pred = predict(w_list[t], train_x)
	# 	acc = accuracy(pred, train_y)
	# 	acc_list[t] = acc
	# 	epochs[t] = t
	# 	t = t + 1
	# plt.plot(np.hstack((0, epochs)), np.hstack((0, acc_list)), label="maximum epochs = {}".format(MAX_EPOCHS))
	# plt.ylabel("accuracy")
	# plt.xlabel("epoch counter")
	# plt.legend()
	# plt.show()
	#################################

	# 1(b)
	# t = 0
	# acc_list = np.zeros(MAX_EPOCHS)
	# acc_list_test = np.zeros(MAX_EPOCHS)
	# epochs = np.zeros(MAX_EPOCHS)
	# w_list = perceptron(MAX_EPOCHS, train_x, train_y)
	# pred_test = None
	# while t < MAX_EPOCHS:
	# 	pred = predict(w_list[t], train_x)
	# 	pred_test = predict(w_list[t], test_x)
	# 	acc = accuracy(pred, train_y)
	# 	acc_test = accuracy(pred_test, test_y)
	# 	acc_list[t] = acc
	# 	acc_list_test[t] = acc_test
	# 	epochs[t] = t
	# 	t = t + 1
	# plt.plot(np.hstack((0, epochs)), np.hstack((0, acc_list)), label="training dataset")
	# plt.plot(np.hstack((0, epochs)), np.hstack((0, acc_list_test)), label="testing dataset")
	# plt.ylabel("accuracy")
	# plt.xlabel("epoch counter")
	# plt.legend()
	# plt.show()
	#################################

	# 1(c)
	# plotConfusionMatrix(pred_test, test_y)
	# print(acc_list_test[t-1])
	#################################
	
	# 1(d)
	# w_list = perceptron_1d(MAX_EPOCHS, train_x, train_y)
	# w_tick = w_list[0]
	# w_star = w_list[1]
	# t = 0
	# b = -500
	# TPR_list_tick = np.zeros(MAX_EPOCHS)
	# FPR_list_tick = np.zeros(MAX_EPOCHS)
	# TPR_list_star = np.zeros(MAX_EPOCHS)
	# FPR_list_star = np.zeros(MAX_EPOCHS)
	# while t < MAX_EPOCHS:
	# 	pred_tick = predict_b(b, w_tick, train_x)
	# 	TPR_list_tick[MAX_EPOCHS-t-1] = getTPR(pred_tick, train_y)
	# 	FPR_list_tick[MAX_EPOCHS-t-1] = getFPR(pred_tick, train_y)

	# 	pred_star = predict_b(b, w_star, train_x)
	# 	TPR_list_star[MAX_EPOCHS-t-1] = getTPR(pred_star, train_y)
	# 	FPR_list_star[MAX_EPOCHS-t-1] = getFPR(pred_star, train_y)
	# 	b = b + 10
	# 	t = t + 1
	# plt.plot(FPR_list_tick, TPR_list_tick, label="w'")
	# plt.plot(FPR_list_star, TPR_list_star, label="w*")
	# plt.ylabel("TPR")
	# plt.xlabel("FPR")
	# plt.legend()
	# plt.show()
	#################################

	# 1(e)
	# area_tick = np.trapz(TPR_list_tick, x = FPR_list_tick)
	# area_star = np.trapz(TPR_list_star, x = FPR_list_star)
	# print("AUC for w': %10f" % area_tick)
	# print("AUC for w*: %10f" % area_star)
	# print(area_star)
	#################################

	# 2(a)
	# training
	# w_list = balancedwinnow(0.1, MAX_EPOCHS, train_x, train_y)
	# t = 0
	# acc_list = np.zeros(MAX_EPOCHS)
	# epochs = np.zeros(MAX_EPOCHS)
	# while t < MAX_EPOCHS:
	# 	pred = predict_winnow(w_list[t], train_x)
	# 	acc = accuracy(pred, train_y)
	# 	acc_list[t] = acc
	# 	epochs[t] = t
	# 	t = t + 1

	# plt.plot(np.hstack((0, epochs)), np.hstack((0, acc_list)), label="training dataset")
	# plt.ylabel("accuracy")
	# plt.xlabel("epoch counter")
	# plt.legend()
	# plt.show()

	# testing
	# pred = predict_winnow(w_list[MAX_EPOCHS-1], test_x)
	# acc = accuracy(pred, test_y)
	# plotConfusionMatrix(pred, test_y)
	# print(acc)
	#################################

	# 2(b)
	# eta_list = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
	# for eta in eta_list:
	# 	w_list = balancedwinnow(eta, MAX_EPOCHS, train_x, train_y)
	# 	t = 0
	# 	acc_list = np.zeros(MAX_EPOCHS)
	# 	epochs = np.zeros(MAX_EPOCHS)
	# 	while t < MAX_EPOCHS:
	# 		pred = predict_winnow(w_list[t], train_x)
	# 		acc = accuracy(pred, train_y)
	# 		acc_list[t] = acc
	# 		epochs[t] = t
	# 		t = t + 1
	# 	plt.plot(np.hstack((0, epochs)), np.hstack((0, acc_list)), label="eta = {}".format(eta))
	# plt.xlabel("epoch counter")
	# plt.ylabel("accuracy")
	# plt.legend()
	# plt.show()


