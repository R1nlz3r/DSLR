import numpy as np
import warnings

def get_data():
	try:
		data = np.genfromtxt("../resources/dataset_test.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_test.csv",\
		 	delimiter = ',', dtype = np.str)
		weights = np.genfromtxt("./theta.csv", delimiter = ',')
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
	return data, data_str, weights

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def decision_boundary(prob):
	prob[prob >= 0.5] = 1
	prob[prob < 0.5] = 0
	return prob

def cpt_db(x):
	return decision_boundary(sigmoid(x))

def save_pred(save):
	indexes = np.zeros((len(save), 1), dtype = long)
	save = save.astype(str)
	for i in range(0, len(save)):
		indexes[i, 0] = i
		if save[i, 0] == '0':
			save[i, 0] = "Ravenclaw"
		elif save[i, 0] == '1':
			save[i, 0] = "Slytherin"
		elif save[i, 0] == '2':
			save[i, 0] = "Gryffindor"
		elif save[i, 0] == '3':
			save[i, 0] = "Hufflepuff"
	save = np.hstack((indexes, save))
	header = ["Index", "Hogwarts House"]
	np.savetxt("houses.csv", save, header = ",".join(header), delimiter = ',',\
		fmt = "%s", comments = "");

def main():
	warnings.simplefilter("ignore")
	data, data_str, weights = get_data()
	for i in range(0, 4):
		if (weights[:, 3] == i).sum() == 0:
			print("Error")
			exit()
	for i in range(1, len(data)):
		exceptions = np.zeros((1, 4))
		labels = np.zeros((1, 4))
		for j in range(0, len(weights)):
			x = data[i, [weights[j, [0]].astype(long), \
				weights[j, [1]].astype(long)]].T.copy()
			if np.isnan(x).any(axis = 1):
				exceptions[0, weights[j, 3].astype(int)] += 1
				continue
			x = np.hstack((np.ones((x.shape[0], 1)), x))
			for k in range(2, weights[j, 2].astype(int) + 1):
				x = np.hstack((x, np.power(x[:, 1:3], k)))
			theta = weights[j, 4:(weights[j, 2].astype(int) * 2 + 5)]
			labels[0, weights[j, 3].astype(int)] += sigmoid(np.dot(x, theta))
		for k in range(0, 4):
			labels[0, k] /= ((weights[:, 3] == k).sum() - exceptions[0, k])
			if np.isnan(labels[0, k]):
				labels[0, k] = 0
		max_pos = np.argmax(labels)
		try:
			save = np.vstack((save, max_pos))
		except:
			save = max_pos
	save_pred(save)

if __name__ == "__main__":
	main()
