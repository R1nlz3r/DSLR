import numpy as np
import matplotlib.pyplot as plt

def get_col(data, data_str, column):
	ravenclaw = np.array([])
	slytherin = np.array([])
	gryffindor = np.array([])
	hufflepuff = np.array([])
	for k in range(1, len(data)):
		if data_str[k, 1] == "Ravenclaw":
			ravenclaw = np.append(ravenclaw, data[k, column])
		elif data_str[k, 1] == "Slytherin":
			slytherin = np.append(slytherin, data[k, column])
		elif data_str[k, 1] == "Gryffindor":
			gryffindor = np.append(gryffindor, data[k, column])
		elif data_str[k, 1] == "Hufflepuff":
			hufflepuff = np.append(hufflepuff, data[k, column])
	return ravenclaw, slytherin, gryffindor, hufflepuff

def main():
	try:
		data = np.genfromtxt("../resources/dataset_train.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_train.csv",\
		 	delimiter = ',', dtype = np.str)
	except:
		print "Error"
		exit()
	colors = ['r', 'b', 'g', 'y']
	for i in range(6, data.shape[1]):
		for j in range(i + 1, data.shape[1]):
			for k in range (0, 4):
				plt.scatter(get_col(data, data_str, i)[k], \
					get_col(data, data_str, j)[k], c = colors[k], \
					edgecolors = "none")
			plt.xlabel(data_str[0, i])
			plt.ylabel(data_str[0, j])
			plt.legend(["Ravenclaw", "Slytherin", "Gryffindor", \
				"Hufflepuff"], fontsize = "small")
			plt.show()

if __name__ == "__main__":
	main()
