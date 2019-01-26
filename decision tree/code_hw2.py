import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# perceptron algorithm from hw1
def perceptron(runs, setx, sety):
	# initialize w and I
	w = np.zeros(setx.shape[1], dtype=np.float64)
	I = runs
	t = 0
	data_size = setx.shape[0]
	w = np.zeros(setx.shape[1])
	done = False
	while t < I:
		done = True
		for i in range(0, data_size):
			if np.dot(w, setx[i]) * sety[i] <= 0:
				w = np.add(w, sety[i] * setx[i])
				done = False
		if done == True:
			print("Number of iterations: %d" % (t+1))
			break
		else:	
			t = t + 1
	return w

# compute gini index given array of labels
def getGiniIndex(nodeLabels):
	if len(nodeLabels)==0:
		return 0
	p = len(list(filter(lambda x: x==1, nodeLabels))) / len(nodeLabels)
	return 2*p*(1-p)

# split the node based on given feature and threshold
def splitNode(feature, threshold, setx, sety):
	left, right, left_label, right_label = list(), list(), list(), list()
	for i in range(0, setx.shape[0]):
		if setx[i][feature] < threshold:
			left.append(setx[i])
			left_label.append(sety[i])
		else:
			right.append(setx[i])
			right_label.append(sety[i])
	return left, right, left_label, right_label

# recursively grow the decision tree
# this only works for the particular problem
# since it assumes decision tree perfectly splitting the dataset exists
# no stopping criteria (n_min) implemented
def decisionTree(setx, sety):
	b_index, b_value, b_score, b_left, b_right, b_left_label, b_right_label = 999, 999, 999, None, None, None, None
	total = len(sety)
	for feature in range(0, setx.shape[1]):
		for row in setx:
			left, right, left_label, right_label = splitNode(feature, row[feature], setx, sety)
			count_l = len(left_label)
			count_r = len(right_label)
			gini = count_l/total*getGiniIndex(left_label) + count_r/total*getGiniIndex(right_label)
			print("X%d < %.3f , sum of children's Gini Index=%.10f" % ((feature+1), row[feature], gini))
			if gini < b_score:
				b_index, b_value, b_score, b_left, b_right, b_left_label, b_right_label = feature, row[feature], gini, left, right, left_label, right_label
	print("Split on X%d < %.3f with sum of children's Gini Index = %.10f" % (b_index+1, b_value, b_score))

	if getGiniIndex(b_left_label) > 0:
		decisionTree(np.array(b_left), np.array(b_left_label))
	if getGiniIndex(b_right_label) > 0:
		decisionTree(np.array(b_right), np.array(b_right_label))

# decision stump for problem 2
def splitStumpNode(feature, threshold, setx, sety):
	left, right, left_label, right_label = list(), list(), list(), list()
	for i in range(0, setx.shape[0]):
		if setx[i][feature] == threshold:
			left.append(setx[i])
			left_label.append(sety[i])
		else:
			right.append(setx[i])
			right_label.append(sety[i])
	return left, right, left_label, right_label

def decisionStump(setx, sety):
	b_index, b_value, b_score, b_left, b_right, b_left_label, b_right_label = 999, 999, 999, None, None, None, None
	total = len(sety)
	for feature in range(0, setx.shape[1]):
		# split on 0
		left, right, left_label, right_label = splitStumpNode(feature, 1, setx, sety)
		count_l = len(left_label)
		count_r = len(right_label)
		gini = count_l/total*getGiniIndex(left_label) + count_r/total*getGiniIndex(right_label)
		#print("X%d == %d , sum of children's Gini Index=%.10f" % ((feature+1), 1, gini))
		if gini < b_score:
			b_index, b_value, b_score, b_left, b_right, b_left_label, b_right_label = feature, 1, gini, left, right, left_label, right_label

	#print("Split on X%d == %d with sum of children's Gini Index = %.10f" % (b_index+1, b_value, b_score))
	#print("p-Left: %.5f" % (len(list(filter(lambda x: x==1, b_left_label))) / len(b_left_label)))
	#print("p-right: %.5f" % (len(list(filter(lambda x: x==1, b_right_label))) / len(b_right_label)))

	return b_score, b_index

def surrogateStump(bestIndex, setx, sety):
	total = len(sety)
	Xj = setx[:,bestIndex]
	PL = len(list(filter(lambda x: x==0, Xj))) / total
	PR = 1 - PL
	minPLPR = PL if PL < PR else PR
	lambda_max = -math.inf 
	surrogate_index = -1
	for feature in range(0, setx.shape[1]):
		if feature == bestIndex:
			continue
		else:
			PLL = len(setx[(setx[:,feature]==0) & (setx[:,bestIndex]==0)]) / total
			PRR = len(setx[(setx[:,feature]==1) & (setx[:,bestIndex]==1)]) / total
			temp = (minPLPR - 1 + PLL + PRR) / minPLPR
			if temp > lambda_max:
				lambda_max = temp
				surrogate_index = feature
	#print("surrogate split on X%d" % (surrogate_index+1))
	left, right, left_label, right_label = splitStumpNode(surrogate_index, 1, setx, sety)
	count_l = len(left_label)
	count_r = len(right_label)
	gini = count_l/total*getGiniIndex(left_label) + count_r/total*getGiniIndex(right_label)
	#print("X%d == %d , sum of children's Gini Index=%.10f" % ((surrogate_index+1), 1, gini))

	return gini, surrogate_index

# compute loss on certain feature
def getLeastSquareError(setx, sety, feature):
	error = 0
	for i in range(0, len(sety)):
		if setx[i][feature] != sety[i]:
			error += 1
	error = error / len(sety)
	return error

# compute mean prediction loss
def getMeanPredictionLoss(prediction, label):
	error = 0
	for i in range(0, len(label)):
		if prediction[i] != label[i]:
			error += 1
	error = error / len(label)
	return error

# normalize the vector on l2-norm
def normalize2(array):
	norm = np.amax(np.linalg.norm(array, axis=1, ord=2))
	return array / norm

if __name__ == '__main__':
	# dataset for problem 1
	# points = np.array([[.75, .10],
	# 				   [.85, .80],
	# 				   [.85, .95],
	# 				   [.15, .10],
	# 				   [.05, .25],
	# 				   [.05, .50],
	# 				   [.85, .25]]);

	# label = np.array([-1, -1, 1, -1, 1, 1, -1]);
	# points_n = normalize2(points)

	# 1(a)
	# w_perceptron = perceptron(100, points_n, label)
	# plt.scatter([.85, .05, .05], [.95, .25, .50], color='b', marker='+')
	# plt.scatter([.75, .85, .15, .85], [.10, .80, .10, .25], color='r', marker='_')
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# plt.plot([0, -1/w_perceptron[0]], [0, 1/w_perceptron[1]], label="perceptron decision boundary")
	# plt.plot([0, .99], [0, 1], label="some other linear decision boundary")
	# plt.plot([0, .95], [0, 1], label="another linear decision boundary")
	# plt.plot([0, .91], [0, 1], label="yet another linear decision boundary")
	# plt.legend()
	# plt.show()

	# 1(b)
	# decisionTree(points, label)
	# plt.scatter([.85, .05, .05], [.95, .25, .50], color='b', marker='+')
	# plt.scatter([.75, .85, .15, .85], [.10, .80, .10, .25], color='r', marker='_')
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# plt.axvline(x=.15, linewidth=1, color='g', label="threshold X1 < 0.15")
	# plt.axhline(y=.95, linewidth=1, color='y', label="threshold X2 < 0.95")
	# plt.legend()
	# plt.show()

	# 1(c)
	# plt.scatter([.85, .05, .05], [.95, .25, .50], color='b', marker='+')
	# plt.scatter([.75, .85, .15, .85], [.10, .80, .10, .25], color='r', marker='_')
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# x = np.arange(0.0, 1.0, 0.01)
	# y = np.sqrt(x)
	# plt.plot(x, y, label="3-point line")
	# z = x * math.sqrt(2)
	# plt.plot(x, z, label="optimal classifier")
	# plt.legend()
	# plt.show()

	# 1(d)
	# Tree1
	# plt.scatter([.85, .05, .05], [.95, .25, .50], color='b', marker='+')
	# plt.scatter([.75, .85, .15, .85], [.10, .80, .10, .25], color='r', marker='_')
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# x = np.arange(0.0, 1.0, 0.01)
	# y = np.sqrt(x)
	# plt.plot(x, y, label="3-point line")
	# plt.axvline(x=.426777, ymin=.46194, ymax=.844623, linewidth=1, color='g', label="decision boundary")
	# plt.axhline(y=.46194, xmax=.426777, linewidth=1, color='g')
	# plt.axhline(y=.844623, xmin=.426777, linewidth=1, color='g')
	# plt.legend()
	# plt.show()

	# Tree2
	# plt.scatter([.85, .05, .05], [.95, .25, .50], color='b', marker='+')
	# plt.scatter([.75, .85, .15, .85], [.10, .80, .10, .25], color='r', marker='_')
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# x = np.arange(0.0, 1.0, 0.01)
	# y = np.sqrt(x)
	# plt.plot(x, y, label="3-point line")
	# plt.axhline(y=.607625, xmin=.0923021, xmax=.646115, linewidth=1, color='g', label="decision boundary")
	# plt.axvline(x=.0923021, ymax=.607625, linewidth=1, color='g')
	# plt.axvline(x=.646115, ymin=.607625, linewidth=1, color='g')
	# plt.legend()
	# plt.show()

	# 1(h)
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# plt.plot([0, 1], [0, 0.288675], color='g', label="optimal linear classifier")
	# plt.axhline(y=.25, xmin=.5, linewidth=1)
	# plt.axvline(x=.5, ymax=.25, linewidth=1)
	# plt.legend()
	# plt.show()

	# 1(i)
	# plt.xlim(0,1)
	# plt.ylim(0,1)
	# plt.xlabel("X1")
	# plt.ylabel("X2")
	# plt.axhline(y=.25, xmin=.5, linewidth=1, label="s1 = 0.25")
	# plt.axvline(x=.5, ymax=.25, linewidth=1, label="s2 = 0.5")
	# plt.legend()
	# plt.show()
	###################################################################

	# dataset for problem 2
	# train_data = np.genfromtxt("train.csv", delimiter=',', skip_header=1)
	# train_x = train_data[:,0:5]
	# train_y = train_data[:,5]

	# test_data = np.genfromtxt("test.csv", delimiter=',', skip_header=1)
	# test_x = test_data[:,0:5]
	# test_y = test_data[:,5]

	# 2(a)
	# (i)
	# gini_split, bestIndex = decisionStump(train_x, train_y)
	# gini_init = getGiniIndex(train_y)
	# # (ii)
	# print("Best split Delta Gini Index = %.10f" % (gini_init - gini_split))

	# gini_surrogate, surrogateIndex = surrogateStump(bestIndex, train_x, train_y)
	# print("Surrogate split Delta Gini Index = %.10f" % (gini_init - gini_surrogate))
	# # (iii)
	# print("Least-square error for best split: %.10f" % getLeastSquareError(test_x, test_y, bestIndex))
	# print("Least-square error for surrogate split: %.10f" % getLeastSquareError(test_x, test_y, surrogateIndex))

	# 2(b)
	# MAX_STUMPS = 1000
	# MAX_SAMPLES = int(0.8 * len(train_x))
	# INDICES_LIST = np.arange(500)
	# np.random.seed(2018)
	# gini_init = getGiniIndex(train_y)

	# best_counter = np.zeros(5)
	# surrogate_counter = np.zeros(5)
	# var_importance = np.zeros(5)
	# var_appearance = np.zeros(5)
	# var_oobimportance = np.zeros(5)
	# prediction_list1 = np.zeros((len(test_y),2))
	# prediction_list2 = np.zeros((MAX_STUMPS*5,len(test_y)))

	# for k in range(1,6):
	# 	for M in range(MAX_STUMPS):
	# 		feature_list = np.sort(np.random.choice(train_x.shape[1], size=k, replace=False))
	# 		samples_list = np.random.choice(train_x.shape[0], size=MAX_SAMPLES, replace=True)
			
	# 		sample_x = (train_x[:,feature_list])[samples_list,:]
	# 		sample_y = train_y[samples_list]

	# 		# count best split and surrogate split
	# 		# compute variable importance using equation (2) and (3)
	# 		b_gini, b_index = decisionStump(sample_x, sample_y)
	# 		best_counter[feature_list[b_index]] += 1
	# 		var_importance[feature_list[b_index]] += gini_init - b_gini
	# 		var_appearance[feature_list[b_index]] += 1
	# 		if k != 1:
	# 			s_gini, s_index = surrogateStump(b_index, sample_x, sample_y)
	# 			surrogate_counter[feature_list[s_index]] += 1
	# 			#var_importance[feature_list[s_index]] += gini_init - s_gini
	# 			#var_appearance[feature_list[s_index]] += 1

	# 		# compute variable importance using equation (5) and (6)
	# 		oob_list = np.delete(INDICES_LIST, samples_list)
	# 		oob_x = train_x[oob_list,:]
	# 		oob_y = train_y[oob_list]

	# 		oob_error = getLeastSquareError(oob_x, oob_y, feature_list[b_index])

	# 		permutation = np.random.permutation(np.arange(len(oob_list)))
	# 		oob_x_perm = oob_x
	# 		for i in range(len(oob_list)):
	# 			oob_x_perm[i][feature_list[b_index]] = oob_x[permutation[i]][feature_list[b_index]]

	# 		oob_error_perm = getLeastSquareError(oob_x_perm, oob_y, feature_list[b_index])
	# 		var_oobimportance[feature_list[b_index]] += oob_error_perm - oob_error

	# 		# compute mean least squares loss using two methods
	# 		for i in range(len(test_y)):
	# 			if test_x[i][feature_list[b_index]] == 1:
	# 				prediction_list1[i][0] += 1
	# 				prediction_list2[(k-1)*MAX_STUMPS+M][i] = 1
	# 			else:
	# 				prediction_list1[i][1] += 1
	# 				prediction_list2[(k-1)*MAX_STUMPS+M][i] = 0



	# print("Number of times a variable is the best split:")
	# print(best_counter)
	# print("Number of times a variable is the surrogate split:")
	# print(surrogate_counter)
	# print("Variable importance(5):")
	# print(var_importance/var_appearance)
	# print("Variable importance(6):")
	# print(var_oobimportance/var_appearance)

	# first_predict = np.zeros(len(test_y))
	# for i in range(len(test_y)):
	# 	first_predict[i] = 1 if prediction_list1[i][0] >= prediction_list1[i][1] else 0

	# second_predict = 0
	# for i in range(len(prediction_list2)):
	# 	second_predict += getMeanPredictionLoss(prediction_list2[i], test_y)
	# second_predict = second_predict / len(prediction_list2)
	# print("Mean least-square loss using first method: %.10f" % getMeanPredictionLoss(first_predict, test_y))
	# print("Mean least-square loss using second method: %.10f" % second_predict)

	# 2(c)
	# MAX_STUMPS = 1000
	# Q_LIST = np.array([.4, .5, .6, .7, .8])
	# INDICES_LIST = np.arange(500)
	# np.random.seed(2018)
	# gini_init = getGiniIndex(train_y)

	# for Q in Q_LIST:
	# 	sample_size = int(Q * len(train_x))
	# 	best_counter = np.zeros(5)
	# 	surrogate_counter = np.zeros(5)
	# 	var_importance = np.zeros(5)
	# 	var_appearance = np.zeros(5)
	# 	var_oobimportance = np.zeros(5)
	# 	var_squares = np.zeros(5)
	# 	var_oobsquares = np.zeros(5)

	# 	for M in range(MAX_STUMPS):
	# 		feature_list = np.sort(np.random.choice(train_x.shape[1], size=2, replace=False))
	# 		samples_list = np.random.choice(train_x.shape[0], size=sample_size, replace=True)
			
	# 		sample_x = (train_x[:,feature_list])[samples_list,:]
	# 		sample_y = train_y[samples_list]

	# 		# count best split and surrogate split
	# 		b_gini, b_index = decisionStump(sample_x, sample_y)
	# 		var_importance[feature_list[b_index]] += gini_init - b_gini
	# 		var_appearance[feature_list[b_index]] += 1
	# 		var_squares[feature_list[b_index]] += (gini_init - b_gini)**2

	# 		# compute variable importance using equation (5) and (6)
	# 		oob_list = np.delete(INDICES_LIST, samples_list)
	# 		oob_x = train_x[oob_list,:]
	# 		oob_y = train_y[oob_list]

	# 		oob_error = getLeastSquareError(oob_x, oob_y, feature_list[b_index])

	# 		permutation = np.random.permutation(np.arange(len(oob_list)))
	# 		oob_x_perm = oob_x
	# 		for i in range(len(oob_list)):
	# 			oob_x_perm[i][feature_list[b_index]] = oob_x[permutation[i]][feature_list[b_index]]

	# 		oob_error_perm = getLeastSquareError(oob_x_perm, oob_y, feature_list[b_index])
	# 		var_oobimportance[feature_list[b_index]] += oob_error_perm - oob_error
	# 		var_oobsquares[[feature_list[b_index]]] += (oob_error_perm - oob_error)**2

	# 	# mean
	# 	print("For B=%d" % sample_size)
	# 	print("Variable importance(5):")
	# 	print(var_importance/var_appearance)
	# 	print("Variable importance(6):")
	# 	print(var_oobimportance/var_appearance)

	# 	# standard deviation
	# 	print("For B=%d" % sample_size)
	# 	print("STD(Variable importance(5)):")
	# 	print(np.sqrt(var_squares/var_appearance - (var_importance/var_appearance)**2))
	# 	print("STD(Variable importance(6)):")
	# 	print(np.sqrt(var_oobsquares/var_appearance - (var_oobimportance/var_appearance)**2))



