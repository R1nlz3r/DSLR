import numpy as np

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

def compute_gradients(x, y, m, theta, house, iterations, alpha):
	x = x[:, [3, 2]] #change data to work with
	y[y != house] = -1
	y[y == house] = 1
	y[y == -1] = 0
	y = y[~np.isnan(x).any(axis = 1)]
	x = x[~np.isnan(x).any(axis = 1)]
	x = np.hstack((np.ones((len(x), 1)), x))
	for i in range(0, iterations):
		theta -= alpha * (np.dot(x.T, (sigmoid(np.dot(x, theta))) - y) / m)
	print("Theta :")
	print(theta)

def main():
	data, data_str = get_data()
	houses = np.where(data_str == "Hogwarts House")[1][0]
	x = data[1:, 5:]
	y = data[1:, [houses]]
	m = len(x)
	theta = np.zeros((3, 1))
	iterations = 200
	alpha = 0.1
	for house in range(0, 4):
		compute_gradients(x, y, m, theta, house, iterations, alpha)

if __name__ == "__main__":
	main()
