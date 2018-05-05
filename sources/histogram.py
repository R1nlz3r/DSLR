import numpy as np
import matplotlib.pyplot as plt

def main():
	try:
		data = np.genfromtxt("../resources/dataset_train.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_train.csv",\
		 	delimiter = ',', dtype = np.str)
	except:
		print "Error"
		exit()
	for i in range(6, data.shape[1]):
		ravenclaw = np.array([])
		slytherin = np.array([])
		gryffindor = np.array([])
		hufflepuff = np.array([])
		for j in range(1, len(data)):
			if data_str[j, 1] == "Ravenclaw":
				ravenclaw = np.append(ravenclaw, data[j, i])
			elif data_str[j, 1] == "Slytherin":
				slytherin = np.append(slytherin, data[j, i])
			elif data_str[j, 1] == "Gryffindor":
				gryffindor = np.append(gryffindor, data[j, i])
			elif data_str[j, 1] == "Hufflepuff":
				hufflepuff = np.append(hufflepuff, data[j, i])
		plt.hist(np.extract(~np.isnan(ravenclaw), ravenclaw), bins = 15, histtype = "stepfilled", alpha = 0.6)
		plt.hist(np.extract(~np.isnan(slytherin), slytherin), bins = 15, histtype = "stepfilled", alpha = 0.6)
		plt.hist(np.extract(~np.isnan(gryffindor), gryffindor), bins = 15, histtype = "stepfilled", alpha = 0.6)
		plt.hist(np.extract(~np.isnan(hufflepuff), hufflepuff), bins = 15, histtype = "stepfilled", alpha = 0.6)
		plt.title(data_str[0, i])
		plt.legend(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"], \
			fontsize = "small")
		plt.show()

if __name__ == "__main__":
	main()
