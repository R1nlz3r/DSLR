import numpy as np
import matplotlib.pyplot as plt
import warnings

def get_data():
	try:
		data = np.genfromtxt("../resources/dataset_train.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_train.csv",\
		 	delimiter = ',', dtype = np.str)
	except:
		print("Error")
		exit()
	best_hand = np.where(data_str == "Best Hand")[1][0]
	data_str[np.where(data_str == "Left")[0], best_hand] = 1
	data_str[np.where(data_str == "Right")[0], best_hand] = 2
	data[1:, best_hand] = data_str[1:, best_hand]
	birthday = np.where(data_str == "Birthday")[1]
	for i in range(1, len(data_str)):
		nb_bday = data_str[i, birthday]
		data[i, birthday] = int(nb_bday[0][:4]) + \
			(int(nb_bday[0][5:7]) / 120 * 10) + \
			(int(nb_bday[0][8:9]) / 3100 * 10)
	houses = np.where(data_str == "Hogwarts House")[1][0]
	data_str[np.where(data_str == "Ravenclaw")[0], houses] = 0
	data_str[np.where(data_str == "Slytherin")[0], houses] = 1
	data_str[np.where(data_str == "Gryffindor")[0], houses] = 2
	data_str[np.where(data_str == "Hufflepuff")[0], houses] = 3
	data[1:, houses] = data_str[1:, houses]
	return data, data_str

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def decision_boundary(prob):
	prob[prob >= 0.5] = 1
	prob[prob < 0.5] = 0
	return prob

def compute_f1(x, y, theta):
	tp = ((y.astype(bool) & (decision_boundary(sigmoid(np.dot(x, theta))) \
		== 1)) == True).sum()
	tn = ((np.invert(y.astype(bool)) & (decision_boundary(sigmoid(np.dot(x, \
		theta))) == 0)) == True).sum()
	fp = ((np.invert(y.astype(bool)) & (decision_boundary(sigmoid(np.dot(x, \
		theta))) == 1)) == True).sum()
	fn = ((y.astype(bool) & (decision_boundary(sigmoid(np.dot(x, theta))) \
		== 0)) == True).sum()
	precision = tp.astype(float) / (tp + fp)
	recall = tp.astype(float) / (tp + fn)
	if np.isnan(precision):
		precision = 0
	if np.isnan(recall):
		recall = 0
	if precision + recall == 0:
		return 0
	f1 = (2 * precision * recall) / (precision + recall)
	if np.isnan(f1):
		return 0
	return f1

def compute_gradients(data, house, iterations, alpha, power, plot = False):
	x = data[1:, 4:].copy()
	x = x[:, [6, 9]] #change features to work with
	y = data[1:, [1]].copy()
	y[y != house] = -1
	y[y == house] = 1
	y[y == -1] = 0
	y = y[~np.isnan(x).any(axis = 1)]
	x = x[~np.isnan(x).any(axis = 1)]
	x = np.hstack((np.ones((x.shape[0], 1)), x))
	for i in range(2, power + 1):
		x = np.hstack((x, np.power(x[:, 1:3], i)))
	theta = np.ones((x.shape[1], 1))
	for i in range(0, len(theta)):
		theta[i] * np.random.randn()
	m = x.shape[0]
	for i in range(0, iterations):
		theta -= alpha * (np.dot(x.T, (sigmoid(np.dot(x, theta)) - y)) / m)
	if (plot == True):
		gridx, gridy = np.mgrid[min(x[:, 1]):max(x[:, 1]):70j, \
			min(x[:, 2]):max(x[:, 2]):70j]
		gridx = gridx.reshape((len(gridx) ** 2, 1))
		gridy = gridy.reshape((len(gridy) ** 2, 1))
		grid = np.hstack((np.ones((len(gridx), 1)), gridx, gridy))
		for i in range(2, power + 1):
			grid = np.hstack((grid, np.power(gridx, i), np.power(gridy, i)))
		plt.scatter(x[np.where(y == 0), 1], x[np.where(y == 0), 2], alpha = 0.3)
		plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], alpha = 0.3)
		plt.scatter(gridx[decision_boundary(sigmoid(np.dot(grid, theta))) == 1], \
			gridy[decision_boundary(sigmoid(np.dot(grid, theta))) == 1], alpha = 0.1)
		plt.scatter(gridx[decision_boundary(sigmoid(np.dot(grid, theta))) == 0], \
			gridy[decision_boundary(sigmoid(np.dot(grid, theta))) == 0], alpha = 0.1)
#		plt.scatter(x[(decision_boundary(sigmoid(np.dot(x, theta))) == 1).any(axis = 1), 1],
#			x[(decision_boundary(sigmoid(np.dot(x, theta))) == 1).any(axis = 1), 2], alpha = 0.3)
	return compute_f1(x, y, theta)

def main():
	warnings.simplefilter("ignore")
	data, data_str = get_data()
	iterations = 10
	alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
	power = [1, 2, 3, 4]
	count = 1
	print("Best F1", "Best alpha", "Best power")
	for house_id in range(0, 4):
		best_score = 0
		best_alpha = alpha[0]
		best_power = power[0]
		for i in range(0, len(alpha)):
			for j in range(0, len(power)):
				tmp_score = compute_gradients(data, house_id, iterations, \
					alpha[i], power[j])
				if (best_score < tmp_score):
					best_score = tmp_score
					best_alpha = alpha[i]
					best_power = power[j]
		print(best_score, best_alpha, best_power)
		plt.figure(count)
		count += 1
		compute_gradients(data, house_id, iterations, best_alpha, best_power, True)
	plt.show()

if __name__ == "__main__":
	main()
