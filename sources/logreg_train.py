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

def get_features(data, feature1, feature2, house_id, power):
	x = data[:, [feature1, feature2]].copy()
	y = data[:, [1]].copy()
	y[y != house_id] = -1
	y[y == house_id] = 1
	y[y == -1] = 0
	y = y[~np.isnan(x).any(axis = 1)]
	x = x[~np.isnan(x).any(axis = 1)]
	x = np.hstack((np.ones((x.shape[0], 1)), x))
	for i in range(2, power + 1):
		x = np.hstack((x, np.power(x[:, 1:3], i)))
	return x, y

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def decision_boundary(prob):
	prob[prob >= 0.5] = 1
	prob[prob < 0.5] = 0
	return prob

def cpt_db(x):
	return decision_boundary(sigmoid(x))

def compute_accuracy(x, y, theta):
	tp = ((y.astype(bool) & (cpt_db(np.dot(x, theta)) == 1)) == True).sum()
	tn = ((np.invert(y.astype(bool)) & (cpt_db(np.dot(x, theta)) == 0)) == True).sum()
	fp = ((np.invert(y.astype(bool)) & (cpt_db(np.dot(x, theta)) == 1)) == True).sum()
	fn = ((y.astype(bool) & (cpt_db(np.dot(x, theta)) == 0)) == True).sum()
	accuracy = (tp + tn).astype(float) / (tp + tn + fp + fn)
	if np.isnan(accuracy):
		return 0
	return accuracy

def plot_data(x, y, theta, power):
	# / ! \ Create a figure before
	gridx, gridy = np.mgrid[min(x[:, 1]):max(x[:, 1]):70j, \
		min(x[:, 2]):max(x[:, 2]):70j]
	gridx = gridx.reshape((len(gridx) ** 2, 1))
	gridy = gridy.reshape((len(gridy) ** 2, 1))
	grid = np.hstack((np.ones((len(gridx), 1)), gridx, gridy))
	for i in range(2, power + 1):
		grid = np.hstack((grid, np.power(gridx, i), np.power(gridy, i)))
	plt.scatter(x[np.where(y == 0), 1], x[np.where(y == 0), 2], alpha = 0.3)
	plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], alpha = 0.3)
	plt.scatter(gridx[cpt_db(np.dot(grid, theta)) == 1], \
		gridy[cpt_db(np.dot(grid, theta)) == 1], alpha = 0.1)
	plt.scatter(gridx[cpt_db(np.dot(grid, theta)) == 0], \
		gridy[cpt_db(np.dot(grid, theta)) == 0], alpha = 0.1)
#	plt.scatter(x[(decision_boundary(sigmoid(np.dot(x, theta))) == 1).any(axis = 1), 1],
#		x[(decision_boundary(sigmoid(np.dot(x, theta))) == 1).any(axis = 1), 2], alpha = 0.3)

def compute_gradients(data, feature1, feature2, house_id, iterations, alpha, \
		power, plot = False):
	x, y = get_features(data, feature1, feature2, house_id, power)
	theta = np.ones((x.shape[1], 1))
	for i in range(0, len(theta)):
		theta[i] * np.random.randn()
	m = x.shape[0]
	for i in range(0, iterations):
		theta -= alpha * (np.dot(x.T, (sigmoid(np.dot(x, theta)) - y)) / m)
	if (plot == True):
		plot_data(x, y, theta, power)
	return compute_accuracy(x, y, theta), theta


def main():
	warnings.simplefilter("ignore")
	data, data_str = get_data()
	iterations = 200
	alpha = [0.01, 0.03, 0.1, 0.3, 1, 3]
	power = [1, 2, 3]
	count = 1
	print("Feature1", "Feature2", "House id", "Best Accuracy", "Best alpha", \
		"Best power", "Theta")
	for house_id in range(0, 4):
		for feature_1 in range(6, data.shape[1] - 4):
			for feature_2 in range(feature_1 + 1, data.shape[1] - 4):
				best_score = best_theta = 0
				best_alpha = alpha[0]
				best_power = power[0]
				for i in range(0, len(alpha)):
					for j in range(0, len(power)):
						tmp_score, tmp_theta = compute_gradients(data, \
							feature_1, feature_2, house_id, iterations, \
							alpha[i], power[j])
						if (best_score < tmp_score):
							best_score = tmp_score
							best_alpha = alpha[i]
							best_power = power[j]
							best_theta = tmp_theta
				if (best_score > 0.98):
					plt.figure(count)
					count += 1
					compute_gradients(data, feature_1, feature_2, house_id, \
						iterations, best_alpha, best_power, plot = True)
					print(feature_1, feature_2, house_id, best_score, \
						best_alpha, best_power, best_theta.T)
					tmp_save = np.zeros((1, max(power) * 2 + 4))
					tmp_save[0, 0] = feature_1
					tmp_save[0, 1] = feature_2
					tmp_save[0, 2] = house_id
					tmp_save[0, 3:best_theta.T.shape[1] + 3] = best_theta.T
					try:
						save = np.vstack((save, tmp_save))
					except:
						save = tmp_save
	np.savetxt("theta.csv", save, delimiter = ',');
	plt.show()

if __name__ == "__main__":
	main()
