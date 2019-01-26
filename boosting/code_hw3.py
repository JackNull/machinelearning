# This is written under Python3
import pandas as pd
import numpy as np
import string
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def remove_punctuation(text):
	return text.replace(string.punctuation, ' ')

def get_numpy_data(dataframe, features, label):
	dataframe["intercept"] = 1
	features = ["intercept"] + features
	feature_matrix = np.array(dataframe[features])
	label_array = np.array(dataframe[label])
	return feature_matrix, label_array

# Estimating conditional probability with logistic sigmoid function
def predict_probability(feature_matrix, coefficients):
	# Take dot product of feature_matrix and coefficients
	result = np.dot(feature_matrix, coefficients)
	# Compute the conditional probability
	predictions = 1./(1.+np.exp(-result))
	return predictions

def feature_derivative(errors, feature):
	# Compute the dot product of errors and feature
	derivative = np.dot(errors, feature)
	return derivative

def compute_log_likelihood(feature_matrix, high_rating, coefficients):
	scores = np.dot(feature_matrix, coefficients)
	sig = 1./(1.+np.exp(-scores))
	lp = np.sum(high_rating*np.log(sig) + (1-high_rating)*np.log(1-sig))
	return lp

def logistic_regression(feature_matrix, high_rating, initial_coefficients, step_size, max_iter):
	coefficients = np.array(initial_coefficients)
	for itr in range(max_iter):
		# Predict conditional probability
		predictions = predict_probability(feature_matrix, coefficients)
		# Compute the errors as y - predictions
		errors = high_rating - predictions
		for j in range(len(coefficients)):
			# Compute the derivative for coefficients[j].
			derivative = feature_derivative(errors, feature_matrix[:,j])
			# add the step size times the derivative to the current coefficient
			coefficients[j] = coefficients[j] + derivative * step_size
		# Checking whether log likelihood is increasing
		if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
			lp = compute_log_likelihood(feature_matrix, high_rating, coefficients)
			# print("iteration %*d: log likelihood of observed labels = %.8f" % (int(np.ceil(np.log10(max_iter))), itr, lp))
	return coefficients

def problem2():
	# Load the dataset
	products = pd.read_csv("amazon_baby_small.csv")

	# print(products.head(10)['name'])
	# print("# of high_rating (5-star) reviews =", len(products[products['rating']>=5]))
	# print("# of low_rating (not 5-star) reviews =", len(products[products['rating']<5]))

	products['high_rating'] = (products['rating'] > 4)

	# Apply text cleaning on the review data
	import json
	with open ("important_words.json", "r") as f:
		important_words = json.load(f)
	important_words = [str(s) for s in important_words]

	products = products.fillna({"review":""})
	products["review_clean"] = products["review"].apply(remove_punctuation)

	for word in important_words:
		products[word] = products["review_clean"].apply(lambda s: s.split().count(word))

	# products["contains_perfect"] = products["perfect"].apply(lambda x: 1 if x>=1 else 0)
	# count = np.sum(list(products["contains_perfect"]))
	# print(count)

	products = products.drop(columns = ["name", "review", "review_clean", "rating"])
	
	# Convert DataFrame to Numpy array
	feature_matrix, high_rating = get_numpy_data(products, important_words, "high_rating")

	coefficients = logistic_regression(feature_matrix, high_rating, initial_coefficients=np.zeros(194), step_size=1e-7, max_iter=301)

	# Predict high_rating
	scores = np.dot(feature_matrix, coefficients)
	class_pred = scores > 0
	print("# of predicted high_rating: %d" % list(class_pred).count(True))

	# Measure accuracy
	mistakes = np.logical_xor(class_pred, high_rating)
	num_mistakes = list(mistakes).count(True)
	accuracy = (len(products) - num_mistakes) / len(products)
	print("-----------------------------------------------------")
	print('# Reviews   correctly classified = %d' % (len(products) - num_mistakes))
	print('# Reviews incorrectly classified = %d' % num_mistakes)
	print('# Reviews total                  = %d' % len(products))
	print("-----------------------------------------------------")
	print('Accuracy = %.2f' % accuracy)

	# Word contribution to high rating
	coefficients = list(coefficients[1:])
	word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
	word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

	print(word_coefficient_tuples[:3])

################################################################################

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
	# Sum the weights of all entries with label +1
	total_weight_positive = sum(data_weights[labels_in_node==1])
	# Weight of mistakes for predicting all -1's is equal to the sum above
	WM_minus = total_weight_positive
	# Sum the weights of all entries with label -1
	total_weight_negative = sum(data_weights[labels_in_node==-1])
	# Weight of mistakes for predicting all 1's is equal to the sum above
	WM_plus = total_weight_negative
	# return the tuple (weight, class_label) representing the lower of the two weights
	#	class_label shoud be {0,1}
	# If the two weights are identical, return (weighted_mistakes_all_positive, +1)
	weight = WM_plus if WM_plus <= WM_minus else WM_minus
	class_label = +1 if WM_plus <= WM_minus else -1
	return weight, class_label

def best_splitting_feature(data, features, target, data_weights):
	# These variables will keep track of the best feature and the corresponding error
	best_feature = None
	best_error = float("+inf")
	num_points = float(len(data))
	# Loop through each feature to consider splitting on that feature
	for feature in features:
		# The left split will have all data points where feature value is 0
		# The right split will have all data points where feature value is 1
		left_split = data[data[feature] == 0]
		right_split = data[data[feature] == 1]

		# Apply the same filtering to data_weights to create left_data_weights, right_data_weights
		left_data_weights = data_weights[data[feature] == 0]
		right_data_weights = data_weights[data[feature] == 1]
		# Calculate the weight of mistakes for the left and right sides
		mistake_left, label_left = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
		mistake_right, label_right = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
		# Compute weighted error
		error = (mistake_left + mistake_right) / sum(data_weights)

		if error < best_error:
			best_feature = feature
			best_error = error

	# return the best feature
	return best_feature

def create_leaf(target_values, data_weights):
	# Create a leaf node
	leaf = {"splitting_feature":None, "is_leaf":True}
	# Computed weight of mistakes
	weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
	leaf["prediction"] = best_class
	return leaf

def weighted_decision_tree_create(data, features, target, data_weights, current_depth=1, max_depth=10):
	remaining_features = features[:] # Make a copy of the features.
	target_values = data[target]
	print("--------------------------------------------------------------------")
	print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
	# Stopping condition 1. Error is 0.
	if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
		print("Stopping condition 1 reached.")
		return create_leaf(target_values, data_weights)
	# Stopping condition 2. No more features.
	if remaining_features == []:
		print("Stopping condition 2 reached.")
		return create_leaf(target_values, data_weights)
	# Additional stopping condition (limit tree depth)
	if current_depth > max_depth:
		print("Reached maximum depth. Stopping for now.")
		return create_leaf(target_values, data_weights)

	# If all the datapoints are the same, splitting_feature will be None. Create a leaf
	splitting_feature = best_splitting_feature(data, features, target, data_weights)
	remaining_features.remove(splitting_feature)
    
	left_split = data[data[splitting_feature] == 0]
	right_split = data[data[splitting_feature] == 1]

	left_data_weights = data_weights[data[splitting_feature] == 0]
	right_data_weights = data_weights[data[splitting_feature] == 1]

	print("Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split)))

	# Create a leaf node if the split is "perfect"
	if len(left_split) == len(data):
		print("Creating leaf node.")
		return create_leaf(left_split[target], data_weights)
	if len(right_split) == len(data):
		print("Creating leaf node.")
		return create_leaf(right_split[target], data_weights)
	# Repeat on left and right subtrees
	left_tree = weighted_decision_tree_create(left_split, remaining_features, target, left_data_weights, current_depth+1, max_depth)
	right_tree = weighted_decision_tree_create(right_split, remaining_features, target, right_data_weights, current_depth+1, max_depth)

	return {"is_leaf":False, "prediction":None, "splitting_feature":splitting_feature, "left":left_tree, "right":right_tree}

def count_nodes(tree):
	if tree["is_leaf"]:
		return 1
	return 1 + count_nodes(tree["left"]) + count_nodes(tree["right"])

def classify(tree, x, annotate=False):
	# If the node is a leaf node
	if tree["is_leaf"]:
		if annotate:
			print("At leaf, predicting %s" % tree["prediction"])
		return tree["prediction"]
	else:
		# Split on feature
		split_feature_value = x[tree["splitting_feature"]]
		if annotate:
			print("Split on %s = %s" % (tree["splitting_feature"], split_feature_value))
		if split_feature_value==0:
			return classify(tree["left"], x, annotate)
		else:
			return classify(tree["right"], x, annotate)

def evaluate_classification_error(tree, data, target):
	# Apply the classify(tree, x) to each row in data
	prediction = np.zeros(len(data))
	for i in range(len(data)):
	 	prediction[i] = classify(tree, data.iloc[i])

	return (prediction != data[target]).sum() / float(len(data))

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
	# start with unweighted data
	alpha = np.array([1.]*len(data))
	weights = []
	tree_stumps = []
	target_values = data[target]

	for t in range(num_tree_stumps):
		print("=====================================================")
		print("Adaboost Iteration %d" % t)
		print("=====================================================")
		# Learn a weighted decision tree stump.
		tree_stumps.append(weighted_decision_tree_create(data, features, target, alpha, max_depth=1))
		# Make prediction
		predictions = np.zeros(len(data))
		for i in range(len(data)):
			predictions[i] = classify(tree_stumps[t], data.iloc[i])

		# Produce a Boolean array indicating whether each data point was correctlt classified
		is_correct = predictions == target_values
		is_wrong = predictions != target_values
		# Compute weighted error
		weighted_error = np.dot(alpha, is_wrong) / np.sum(alpha) 
		# Compute model coefficient using weighted error
		weight = 1/2*math.log((1-weighted_error)/weighted_error)
		# Adjust weights on data points
		adjustment = is_correct.apply(lambda is_correct: math.exp(-weight) if is_correct else math.exp(weight))
		# Scale alpha by multiplying by adjustment then normalize data points weights
		alpha = alpha * adjustment
		alpha = alpha / np.sum(alpha)
		weights.append(weight)

	return weights, tree_stumps

def print_stump(tree):
	split_name = tree["splitting_feature"]
	if split_name is None:
		print("(leaf, label: %s)" % tree["prediction"])
		return None
	split_feature, split_value = split_name.split("_")
	print('                       root')
	print('         |---------------|----------------|')
	print('         |                                |')
	print('         |                                |')
	print('         |                                |')
	print('  [{0} == 0]{1}[{0} == 1]    '.format(split_name, ' '*(27-len(split_name))))
	print('         |                                |')
	print('         |                                |')
	print('         |                                |')
	print('    (%s)                 (%s)' \
		% (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
		('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree')))

def predict_adaboost(stump_weights, tree_stumps, data):
	scores = np.array([0.]*len(data))

	for i, tree_stump in enumerate(tree_stumps):
		predictions = data.apply(lambda x: classify(tree_stump, x), axis=1)
		scores = scores + stump_weights[i]*predictions
	return scores.apply(lambda score: +1 if score > 0 else -1)

def problem3():
	# Load the dataset
	loans = pd.read_csv("loan_small.csv")

	loans["safe_loans"] = loans["loan_status"].apply(lambda x: +1 if x=="Fully Paid" else -1)
	loans.drop(columns=["loan_status"], inplace=True)

	target = "safe_loans"

	# transform into binary features
	loans = pd.get_dummies(loans)
	features = list(loans.columns)
	features.remove("safe_loans")

	train_data, test_data = train_test_split(loans, test_size=0.2, random_state=1)

	# Weighted decision trees

	example_data_weights = np.array([1.0 for i in range(len(train_data))])
	small_data_decision_tree = weighted_decision_tree_create(train_data, features, target,
	                                        example_data_weights, max_depth=2)
	if count_nodes(small_data_decision_tree) == 7:
	    print('Test passed!')
	else:
	    print('Test failed... try again!')
	    print('Number of nodes found: %d' % count_nodes(small_data_decision_tree))
	    print('Number of nodes that should be there: 7')

	print(small_data_decision_tree)
	
	score = evaluate_classification_error(small_data_decision_tree, test_data, target)
	print(score)

	# Assign weights
	example_data_weights = np.array([1.] * 10 + [0.]*(len(train_data) - 20) + [1.] * 10)

	# # Train a weighted decision tree model.
	small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, features, target,
	                         example_data_weights, max_depth=2)
	subset_20 = train_data.head(10).append(train_data.tail(10))
	score1 = evaluate_classification_error(small_data_decision_tree_subset_20, subset_20, target)
	print(score1)
	score2 = evaluate_classification_error(small_data_decision_tree_subset_20, train_data, target)
	print(score2)

	stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=2)
	print_stump(tree_stumps[0])
	print_stump(tree_stumps[1])
	print(stump_weights)

	stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=10)
	predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
	accuracy = accuracy_score(test_data[target], predictions)
	print("Accuracy of 10-component ensemble = %s" % accuracy)
	print("Stump weights:")
	print(stump_weights)

	stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=30)
	error_all = []
	for n in range(1, 31):
		predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
		error = 1.0 - accuracy_score(train_data[target], predictions)
		error_all.append(error)
		print("Iteration %s, training error = %s" % (n, error_all[n-1]))

	test_error_all = []
	for n in range(1, 31):
		predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], test_data)
		error = 1.0 - accuracy_score(test_data[target], predictions)
		test_error_all.append(error)
		print("Iteration %s, test error = %s" % (n, test_error_all[n-1]))

	plt.rcParams["figure.figsize"] = 7, 5
	plt.plot(range(1, 31), error_all, '-', linewidth=4.0, label="Training error")
	plt.plot(range(1, 31), test_error_all, '-', linewidth=4.0, label="Test error")
	plt.title("Performance of Adaboost ensemble")
	plt.xlabel("# of iterations")
	plt.ylabel("Classification error")
	plt.legend(loc="best", prop={"size":15})
	plt.rcParams.update({"font.size":16})
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	#problem2()
	#problem3()



	



