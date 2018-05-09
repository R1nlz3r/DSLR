import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tk

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

def get_data():
	try:
		data = np.genfromtxt("../resources/dataset_train.csv", delimiter = ',')
		data_str = np.genfromtxt("../resources/dataset_train.csv",\
		 	delimiter = ',', dtype = np.str)
	except:
		print("Error")
		exit()
	return data, data_str

def set_ticks(data, data_str, axarr, i, j, houses_x, houses_y):
	if i != data.shape[1] - 1:
		axarr[i - 6, j - 6].xaxis.set_minor_locator(tk.NullLocator())
		axarr[i - 6, j - 6].xaxis.set_major_locator(tk.NullLocator())
	else:
		concat = np.concatenate(houses_x)
		axarr[i - 6, (j - 6) * -1 + data.shape[1] - 7].set_xlim([min(concat), \
			max(concat)])
		axarr[i - 6, (j - 6) * -1 + data.shape[1] - 7].tick_params(labelsize = \
			"xx-small")
		axarr[i - 6, (j - 6) * -1 + data.shape[1] - 7].set_xlabel(\
			data_str[0, j], fontsize = 6.5)
	if j != 6:
		axarr[i - 6, j - 6].yaxis.set_minor_locator(tk.NullLocator())
		axarr[i - 6, j - 6].yaxis.set_major_locator(tk.NullLocator())
	else:
		if i == data.shape[1] - 1:
			axarr[i - 6, j - 6].yaxis.set_minor_locator(tk.NullLocator())
			axarr[i - 6, j - 6].yaxis.set_major_locator(tk.NullLocator())
		else:
			concat = np.concatenate(houses_y)
			axarr[i - 6, 0].set_ylim([min(concat), max(concat)])
			axarr[i - 6, 0].tick_params(labelsize = "xx-small")
		axarr[i - 6, 0].set_ylabel(data_str[0, i], fontsize = 5.5)

def main():
	data, data_str = get_data()
	fig, axarr = plt.subplots(data.shape[1] - 6, data.shape[1] - 6)
	plt.subplots_adjust(wspace = 0, hspace = 0, left = 0.07, right = 0.92, \
		top = 0.95, bottom = 0.05)
	colors = ['r', 'b', 'g', 'y']
	for i in range(6, data.shape[1]):
		for j in range(data.shape[1] - 1, 5, -1):
			houses_x = get_col(data, data_str, j)
			houses_y = get_col(data, data_str, i)
			if i != j:
				for k in range(0, 4):
					axarr[i - 6, (j - 6) * -1 + data.shape[1] - 7].scatter(\
						houses_x[k], houses_y[k], c = colors[k], \
						edgecolors = "none", s = 2)
			else:
				concat = np.concatenate(houses_x)
				axarr[i - 6, (j - 6) * -1 + data.shape[1] - 7].hist( \
					np.extract(~np.isnan(concat), concat), bins = 40)
			set_ticks(data, data_str, axarr, i , j, houses_x, houses_y)
	plt.legend(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"], \
		fontsize = "x-small", loc = (1.04, 0))
	plt.show()

if __name__ == "__main__":
	main()
