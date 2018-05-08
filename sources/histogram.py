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
	ravenclaw = np.extract(~np.isnan(ravenclaw), ravenclaw)
	slytherin = np.extract(~np.isnan(slytherin), slytherin)
	gryffindor = np.extract(~np.isnan(gryffindor), gryffindor)
	hufflepuff = np.extract(~np.isnan(hufflepuff), hufflepuff)
	return ravenclaw, slytherin, gryffindor, hufflepuff

def main():
	try:
		data = np.genfromtxt("../resources/dataset_train.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_train.csv",\
		 	delimiter = ',', dtype = np.str)
	except:
		print("Error")
		exit()
	for i in range(6, data.shape[1]):
		plt.hist(get_col(data, data_str, i), bins = 15, \
			histtype = "stepfilled", alpha = 0.8, color = ['y', 'g', 'b', 'r'])
		plt.title(data_str[0, i])
		plt.legend(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"], \
			fontsize = "small")
		plt.show()

if __name__ == "__main__":
	main()
